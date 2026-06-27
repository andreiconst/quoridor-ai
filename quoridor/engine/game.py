"""Fixed-size action-space wrapper around State, for use by MCTS/NN code."""

from __future__ import annotations

import numpy as np

from .state import BOARD_SIZE, WALL_GRID, WALL_H, WALL_V, State

NUM_CELLS = BOARD_SIZE * BOARD_SIZE
NUM_WALL_SLOTS = WALL_GRID * WALL_GRID
ACTION_SIZE = NUM_CELLS + 2 * NUM_WALL_SLOTS  # 81 + 64 + 64 = 209


def pawn_action(dest) -> int:
    r, c = dest
    return r * BOARD_SIZE + c


def wall_action(r: int, c: int, orientation: int) -> int:
    base = NUM_CELLS if orientation == WALL_H else NUM_CELLS + NUM_WALL_SLOTS
    return base + r * WALL_GRID + c


def decode_action(action: int):
    """Returns ('pawn', (r, c)) or ('wall', (r, c, orientation))."""
    if action < NUM_CELLS:
        return "pawn", (action // BOARD_SIZE, action % BOARD_SIZE)
    action -= NUM_CELLS
    if action < NUM_WALL_SLOTS:
        return "wall", (action // WALL_GRID, action % WALL_GRID, WALL_H)
    action -= NUM_WALL_SLOTS
    return "wall", (action // WALL_GRID, action % WALL_GRID, WALL_V)


def legal_action_mask(state: State) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for dest in state.legal_pawn_destinations():
        mask[pawn_action(dest)] = 1.0
    for r, c, orientation in state.legal_wall_slots():
        mask[wall_action(r, c, orientation)] = 1.0
    return mask


def apply_action(state: State, action: int) -> State:
    kind, payload = decode_action(action)
    next_state = state.clone()
    player = next_state.current_player
    if kind == "pawn":
        next_state.move_pawn(player, payload)
    else:
        r, c, orientation = payload
        next_state.place_wall(player, r, c, orientation)
    return next_state


def encode_state(state: State) -> np.ndarray:
    """Encode the board as planes, from the current player's perspective.

    Planes: own pawn, opponent pawn, horizontal walls, vertical walls,
    own walls remaining (constant), opponent walls remaining (constant).
    """
    player = state.current_player
    opp = 1 - player

    def maybe_flip(r, c):
        # Flip the board vertically so the side-to-move always advances
        # "downward" (row 0 -> row 8), which keeps the NN's task symmetric.
        return (BOARD_SIZE - 1 - r, c) if player == 1 else (r, c)

    planes = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    pr, pc = maybe_flip(*state.pawns[player])
    planes[0, pr, pc] = 1.0
    orr, orc = maybe_flip(*state.pawns[opp])
    planes[1, orr, orc] = 1.0

    for r in range(WALL_GRID):
        for c in range(WALL_GRID):
            slot = state.wall_slots[r][c]
            if slot == WALL_H:
                fr, fc = maybe_flip(r, c)
                planes[2, fr, fc] = 1.0
            elif slot == WALL_V:
                fr, fc = maybe_flip(r, c)
                planes[3, fr, fc] = 1.0

    planes[4, :, :] = state.walls_left[player] / 10.0
    planes[5, :, :] = state.walls_left[opp] / 10.0
    return planes


def encode_action_for_player(state: State, action: int) -> int:
    """Re-express an action computed in canonical (flipped) space back to
    the real board, accounting for the perspective flip in encode_state."""
    if state.current_player == 0:
        return action
    kind, payload = decode_action(action)
    if kind == "pawn":
        r, c = payload
        return pawn_action((BOARD_SIZE - 1 - r, c))
    r, c, orientation = payload
    return wall_action(WALL_GRID - 1 - r, c, orientation)
