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
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def _evaluate(self, state: State):
        """Returns (canonical_policy_probs over real actions, value, mask).

        The legal-action mask is the single most expensive thing to compute
        (it path-checks every candidate wall), so we return it here and reuse
        it in _expand rather than recomputing it.
        """
        planes = torch.from_numpy(encode_state(state)).to(self.device)
        probs, value = self.network.predict(planes)
        probs = probs.cpu().numpy()
        mask = legal_action_mask(state)
        # probs are indexed in canonical (possibly flipped) action space;
        # remap each to the corresponding real action index.
        real_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a in np.nonzero(mask)[0]:
            canon_a = encode_action_for_player(state, int(a))
            real_probs[a] = probs[canon_a]
        total = real_probs.sum()
        if total > 0:
            real_probs /= total
        else:
            real_probs = mask / max(mask.sum(), 1)
        return real_probs, value, mask

    def run(self, root_state: State, add_noise: bool = True) -> "Node":
        root = Node(root_state.clone())
        priors, _, mask = self._evaluate(root.state)
        self._expand(root, priors, mask)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            path = [node]
            while node.expanded() and not node.state.is_terminal():
                action, node = self._select_child(node)
                path.append(node)

            leaf = path[-1]
            if leaf.state.is_terminal():
                winner = leaf.state.winner
                value = 1.0 if winner == leaf.player else -1.0
            else:
                priors, value, mask = self._evaluate(leaf.state)
                self._expand(leaf, priors, mask)

            self._backpropagate(path, value)
        return root

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
