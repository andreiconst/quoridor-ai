"""Reader for Go .qsh shards parses the format correctly, and the learner
trains from them and publishes weights atomically."""

import struct
from pathlib import Path

import numpy as np

from quoridor.serving.go_shards import MAGIC, load_go_dir, load_go_shard


def _write_qsh(path, planes, policies, values):
    n = len(values)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, n, 486, 209))
        for i in range(n):
            f.write(planes[i].astype("<f4").tobytes())
            f.write(policies[i].astype("<f4").tobytes())
            f.write(struct.pack("<f", float(values[i])))


def test_reader_parses(tmp_path):
    rng = np.random.default_rng(0)
    planes = rng.random((5, 6, 9, 9), dtype=np.float32)
    policies = rng.random((5, 209), dtype=np.float32)
    values = rng.random(5).astype(np.float32) * 2 - 1
    _write_qsh(tmp_path / "go_000000.qsh", planes, policies, values)

    p, po, v = load_go_shard(str(tmp_path / "go_000000.qsh"))
    assert p.shape == (5, 6, 9, 9) and po.shape == (5, 209) and v.shape == (5,)
    assert np.allclose(p, planes) and np.allclose(po, policies) and np.allclose(v, values)


def test_load_dir_concatenates(tmp_path):
    for s in range(2):
        planes = np.zeros((3, 6, 9, 9), np.float32)
        policies = np.zeros((3, 209), np.float32)
        values = np.full(3, s, np.float32)
        _write_qsh(tmp_path / f"go_{s:06d}.qsh", planes, policies, values)
    p, po, v = load_go_dir(str(tmp_path))
    assert p.shape[0] == 6 and v.shape[0] == 6


def test_learner_trains_and_publishes(tmp_path):
    from quoridor.serving.learner import train_once

    rng = np.random.default_rng(1)
    planes = rng.random((40, 6, 9, 9), dtype=np.float32)
    policies = np.full((40, 209), 1.0 / 209, dtype=np.float32)
    values = rng.random(40).astype(np.float32) * 2 - 1
    _write_qsh(tmp_path / "go_000000.qsh", planes, policies, values)

    out = tmp_path / "current.pt"
    train_once(str(tmp_path), str(out), channels=16, blocks=2, steps=5, batch_size=16, device="cpu")
    assert out.exists()
    assert not (tmp_path / "current.pt.tmp").exists()  # atomic: tmp cleaned up
