"""PUCT MCTS guided by a policy/value network."""

from __future__ import annotations

import math

import numpy as np
import torch

from ..engine.game import (
    ACTION_SIZE,
    apply_action,
    encode_action_for_player,
    encode_state,
    legal_action_mask,
)
from ..engine.state import State

C_PUCT = 1.5


class Node:
    __slots__ = ("state", "prior", "children", "visit_count", "value_sum", "player")

    def __init__(self, state: State, prior: float = 0.0):
        self.state = state
        self.prior = prior
        self.children: dict[int, "Node"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.player = state.current_player

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    def __init__(self, network, device="cpu", num_simulations: int = 200,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 batch_size: int = 16, virtual_loss: float = 1.0):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        # Leaf-parallel batching: collect up to `batch_size` leaves (guided away
        # from each other by a temporary virtual loss) and evaluate them in a
        # single network forward pass.
        self.batch_size = max(1, batch_size)
        self.virtual_loss = virtual_loss

    def _mask_probs(self, state: State, probs: np.ndarray):
        """Map network policy (canonical action space) onto a normalized
        distribution over this state's real legal actions; return (probs, mask)."""
        mask = legal_action_mask(state)
        real_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a in np.nonzero(mask)[0]:
            real_probs[a] = probs[encode_action_for_player(state, int(a))]
        total = real_probs.sum()
        if total > 0:
            real_probs /= total
        else:
            real_probs = mask / max(mask.sum(), 1)
        return real_probs, mask

    def _evaluate(self, state: State):
        """Single-state evaluation. Returns (real_probs, value, mask)."""
        planes = torch.from_numpy(encode_state(state)).to(self.device)
        probs, value = self.network.predict(planes)
        real_probs, mask = self._mask_probs(state, probs.cpu().numpy())
        return real_probs, value, mask

    def _evaluate_batch(self, states: list):
        """Evaluate several states in one network forward pass.

        Returns a list of (real_probs, value, mask) aligned with `states`."""
        planes = np.stack([encode_state(s) for s in states])
        tensor = torch.from_numpy(planes).to(self.device)
        probs, values = self.network.predict(tensor)
        probs = probs.cpu().numpy()
        values = values.cpu().numpy()
        results = []
        for i, state in enumerate(states):
            real_probs, mask = self._mask_probs(state, probs[i])
            results.append((real_probs, float(values[i]), mask))
        return results

    def run(self, root_state: State, add_noise: bool = True) -> "Node":
        root = Node(root_state.clone())
        priors, _, mask = self._evaluate(root.state)
        self._expand(root, priors, mask)
        if add_noise:
            self._add_dirichlet_noise(root)
        if not root.children:
            return root

        sims_done = 0
        while sims_done < self.num_simulations:
            leaves, paths = [], []
            pending = set()
            target = min(self.batch_size, self.num_simulations - sims_done)
            # Gather a batch of distinct, non-terminal leaves to evaluate.
            while len(leaves) < target:
                node = root
                path = [node]
                while node.expanded() and not node.state.is_terminal():
                    _, node = self._select_child(node)
                    path.append(node)
                leaf = path[-1]
                if leaf.state.is_terminal():
                    winner = leaf.state.winner
                    value = 1.0 if winner == leaf.player else -1.0
                    self._backpropagate(path, value)
                    sims_done += 1
                    if sims_done >= self.num_simulations:
                        break
                    continue
                if id(leaf) in pending:
                    # Collision: virtual loss didn't steer us away (e.g. only one
                    # leaf left). Evaluate what we have rather than spin.
                    break
                self._apply_virtual_loss(path)
                pending.add(id(leaf))
                leaves.append(leaf)
                paths.append(path)

            if not leaves:
                continue

            for (priors, value, mask), leaf, path in zip(
                self._evaluate_batch([leaf.state for leaf in leaves]), leaves, paths
            ):
                self._revert_virtual_loss(path)
                self._expand(leaf, priors, mask)
                self._backpropagate(path, value)
                sims_done += 1
        return root

    def _apply_virtual_loss(self, path: list) -> None:
        # Temporarily count each node on the path as a loss so concurrent
        # selections in this batch are discouraged from re-walking it.
        vl = self.virtual_loss
        for node in path:
            node.visit_count += vl
            node.value_sum -= vl

    def _revert_virtual_loss(self, path: list) -> None:
        vl = self.virtual_loss
        for node in path:
            node.visit_count -= vl
            node.value_sum += vl

    def _select_child(self, node: Node):
        best_score, best_action, best_child = -float("inf"), None, None
        sqrt_total = math.sqrt(max(node.visit_count, 1))
        for action, child in node.children.items():
            q = -child.value()  # child stores value for its own player to move
            u = C_PUCT * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def _expand(self, node: Node, priors: np.ndarray, mask: np.ndarray) -> None:
        legal_actions = np.nonzero(mask)[0]
        for action in legal_actions:
            child_state = apply_action(node.state, int(action))
            node.children[int(action)] = Node(child_state, prior=float(priors[action]))

    def _add_dirichlet_noise(self, root: Node) -> None:
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * n

    def _backpropagate(self, path: list, leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    @staticmethod
    def policy_from_visits(root: Node, temperature: float = 1.0) -> np.ndarray:
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)
        if temperature <= 1e-3:
            best = actions[int(np.argmax(visits))]
            policy[best] = 1.0
            return policy
        visits = visits ** (1.0 / temperature)
        visits /= visits.sum()
        for a, p in zip(actions, visits):
            policy[a] = p
        return policy
