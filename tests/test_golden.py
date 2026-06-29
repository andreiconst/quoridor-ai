"""Golden vectors must be reconstructable from their raw state fields and
reproduce the stored planes/masks. This is the same reconstruction the Go port
performs, so it guards cross-language parity at the Python boundary."""

import json
from pathlib import Path

import numpy as np
import pytest

from quoridor.engine.game import encode_state, legal_action_mask
from quoridor.engine.state import State

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "golden.json"


def _state_from_case(case):
    return State(
        pawns=[tuple(case["pawns"][0]), tuple(case["pawns"][1])],
        walls_left=list(case["walls_left"]),
        wall_slots=[list(row) for row in case["wall_slots"]],
        current_player=case["current_player"],
    )


@pytest.mark.skipif(not GOLDEN.exists(), reason="run scripts/dump_golden.py first")
def test_golden_roundtrip():
    cases = json.loads(GOLDEN.read_text())
    assert len(cases) > 50
    for i, case in enumerate(cases):
        state = _state_from_case(case)
        planes = encode_state(state)
        mask = legal_action_mask(state)
        assert np.array_equal(planes, np.array(case["planes"], dtype=np.float32)), f"planes mismatch case {i}"
        assert np.array_equal(mask, np.array(case["mask"], dtype=np.float32)), f"mask mismatch case {i}"
