"""Self-play game generation producing (state_planes, policy, outcome) examples."""

from __future__ import annotations

import numpy as np

from ..engine.game import ACTION_SIZE, encode_action_for_player, encode_state, legal_action_mask
from ..engine.state import State
from .mcts import MCTS

MAX_MOVES = 200
TEMPERATURE_MOVES = 20  # after this many plies, play greedily (temperature -> 0)


def play_one_game(network, device="cpu", num_simulations: int = 200):
    """Returns a list of (planes, policy, player) tuples and the winner."""
    state = State.initial()
    mcts = MCTS(network, device=device, num_simulations=num_simulations)
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
    training_examples = []
    for planes, policy, player in examples:
        if winner == -1:
            outcome = 0.0
        else:
            outcome = 1.0 if player == winner else -1.0
        training_examples.append((planes, policy, outcome))
    return training_examples, winner
