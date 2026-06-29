"""Phase 0: emit golden reference vectors for the Go engine port.

For a set of canonical Quoridor states we record the exact inputs a Go port must
reproduce bit-for-bit:
  - the raw state (pawns, wall_slots, walls_left, current_player) so Go can
    reconstruct the identical position,
  - encode_state planes (6x9x9 float32, the network input),
  - the legal-action mask (209 float32).

Outputs golden/golden_vectors.npz (for Python tests) and golden/golden.json
(nested arrays, easy to load from Go). See docs/PROTOCOL.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quoridor.engine.game import apply_action, legal_action_mask, encode_state
from quoridor.engine.state import State

OUT_DIR = Path(__file__).resolve().parents[1] / "golden"


def _scripted_states(seed: int = 0, n_steps: int = 40):
    """Play a seeded sequence of legal moves, snapshotting along the way so the
    golden set spans walls, jumps, player-1-to-move (flip), and late game."""
    rng = np.random.default_rng(seed)
    state = State.initial()
    snapshots = [state.clone()]
    for _ in range(n_steps):
        mask = legal_action_mask(state)
        actions = np.nonzero(mask)[0]
        if len(actions) == 0 or state.is_terminal():
            break
        action = int(rng.choice(actions))
        state = apply_action(state, action)
        snapshots.append(state.clone())
    return snapshots


def _handcrafted_states():
    cases = []

    # Adjacent pawns -> straight jump available.
    s = State.initial()
    s.pawns = [(3, 4), (4, 4)]
    s.current_player = 0
    cases.append(s)

    # Adjacent pawns with wall behind -> diagonal jumps.
    s = State.initial()
    s.pawns = [(3, 4), (4, 4)]
    s.wall_slots[4][3] = 1  # WALL_H
    s.wall_slots[4][4] = 1
    s.current_player = 0
    cases.append(s)

    # Player 1 to move (exercises the perspective flip in encode_state).
    s = State.initial()
    s.pawns = [(2, 1), (6, 7)]
    s.current_player = 1
    s.walls_left = [7, 3]
    cases.append(s)

    # Near-goal with some walls.
    s = State.initial()
    s.pawns = [(7, 4), (1, 2)]
    s.wall_slots[0][0] = 2  # WALL_V
    s.wall_slots[3][3] = 1  # WALL_H
    s.walls_left = [5, 6]
    s.current_player = 0
    cases.append(s)

    return cases


def build_cases():
    states = _scripted_states(seed=0) + _scripted_states(seed=7) + _handcrafted_states()
    cases = []
    for s in states:
        cases.append({
            "pawns": [list(s.pawns[0]), list(s.pawns[1])],
            "wall_slots": [row[:] for row in s.wall_slots],
            "walls_left": list(s.walls_left),
            "current_player": s.current_player,
            "planes": encode_state(s).tolist(),
            "mask": legal_action_mask(s).tolist(),
        })
    return cases


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = build_cases()

    # JSON (cross-language, Go-friendly).
    (OUT_DIR / "golden.json").write_text(json.dumps(cases))

    # NPZ (stacked arrays, for fast Python-side assertions).
    planes = np.array([c["planes"] for c in cases], dtype=np.float32)
    masks = np.array([c["mask"] for c in cases], dtype=np.float32)
    pawns = np.array([c["pawns"] for c in cases], dtype=np.int64)
    walls = np.array([c["wall_slots"] for c in cases], dtype=np.int64)
    walls_left = np.array([c["walls_left"] for c in cases], dtype=np.int64)
    players = np.array([c["current_player"] for c in cases], dtype=np.int64)
    np.savez_compressed(
        OUT_DIR / "golden_vectors.npz",
        planes=planes, masks=masks, pawns=pawns, wall_slots=walls,
        walls_left=walls_left, current_player=players,
    )

    print(f"Wrote {len(cases)} golden cases to {OUT_DIR}/")
    print(f"  golden.json          ({(OUT_DIR / 'golden.json').stat().st_size/1024:.0f} KB)")
    print(f"  golden_vectors.npz   planes={planes.shape} masks={masks.shape}")


if __name__ == "__main__":
    main()
