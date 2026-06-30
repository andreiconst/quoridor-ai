"""Self-play game generation producing (state_planes, policy, outcome) examples."""

from __future__ import annotations

import numpy as np

from ..engine.game import ACTION_SIZE, encode_action_for_player, encode_state, legal_action_mask
from ..engine.state import State
from .mcts import MCTS

MAX_MOVES = 200  # capped games are resolved by goal-distance, not scored as
# draws (see _winner_by_progress) -- this breaks the cold-start draw trap.
TEMPERATURE_MOVES = 20  # after this many plies, play greedily (temperature -> 0)


def play_one_game(network=None, device="cpu", num_simulations: int = 200,
                  mcts_batch_size: int = 16, evaluator=None):
    """Returns a list of (planes, policy, player) tuples and the winner.

    Pass either a local `network` or a (possibly remote) `evaluator`."""
    state = State.initial()
    mcts = MCTS(network, device=device, num_simulations=num_simulations,
                batch_size=mcts_batch_size, evaluator=evaluator)
    examples = []

    for ply in range(MAX_MOVES):
        if legal_action_mask(state).sum() == 0:
            break
        root = mcts.run(state, add_noise=True)
        temperature = 1.0 if ply < TEMPERATURE_MOVES else 0.1
        policy = MCTS.policy_from_visits(root, temperature=temperature)
        # encode_state flips the board for player 1; remap the policy (which
        # is over real action indices) into that same canonical space so
        # planes and targets line up for training.
        canonical_policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        for real_action in np.nonzero(policy)[0]:
            canon_action = encode_action_for_player(state, int(real_action))
            canonical_policy[canon_action] = policy[real_action]
        examples.append((encode_state(state), canonical_policy, state.current_player))

        action = int(np.random.choice(len(policy), p=policy))
        state = root.children[action].state
        if state.is_terminal():
            break

    winner = state.winner
    if winner == -1:
        # Move cap reached: decide by progress so games are decisive. Without
        # this, weak early self-play wanders to a draw, the value signal stays
        # ~0, and the net never learns to reach its goal (cold-start trap).
        winner = _winner_by_progress(state)
    training_examples = []
    for planes, policy, player in examples:
        if winner == -1:
            outcome = 0.0
        else:
            outcome = 1.0 if player == winner else -1.0
        training_examples.append((planes, policy, outcome))
    return training_examples, winner


def _dist_to_goal(state, player):
    """Shortest path length from the player's pawn to its goal row (BFS,
    respecting walls); a large number if somehow unreachable."""
    from collections import deque

    goal = state.goal_row(player)
    start = state.pawns[player]
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        if r == goal:
            return d
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if state._in_bounds(nr, nc) and (nr, nc) not in seen and not state._edge_blocked(r, c, nr, nc):
                seen.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return 999


def _winner_by_progress(state):
    """Whoever is closer to their goal wins a capped game; equal distance = draw."""
    d0 = _dist_to_goal(state, 0)
    d1 = _dist_to_goal(state, 1)
    if d0 < d1:
        return 0
    if d1 < d0:
        return 1
    return -1
