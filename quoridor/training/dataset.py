"""Persist and reload self-play training examples.

Each example is (planes, policy, value):
  - planes: the encoded board (network input), float16
  - policy: MCTS visit distribution over the 209 actions (the improved policy
            target), float16
  - value:  game outcome from the side-to-move's perspective in {-1, 0, 1}

These tuples are architecture-independent supervised labels, so they double as
(a) a durable, valuable record of all self-play, and (b) a dataset to warm-start
a *new* network via supervised pretraining (see pretrain.py).

Stored as compressed .npz shards. float16 + the heavy zero-sparsity of the
planes/policy compress well (expect very roughly ~0.2-0.4 KB/example on disk).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ShardWriter:
    """Accumulate examples and flush fixed-size compressed shards."""

    def __init__(self, data_dir, shard_size: int = 50_000, prefix: str = "selfplay"):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self._planes: list = []
        self._policies: list = []
        self._values: list = []
        # Continue numbering after any shards already present.
        self._shard_idx = len(list(self.dir.glob(f"{prefix}_*.npz")))
        self.total_written = 0

    def add(self, planes, policy, value) -> None:
        self._planes.append(np.asarray(planes, dtype=np.float16))
        self._policies.append(np.asarray(policy, dtype=np.float16))
        self._values.append(np.float16(value))
        if len(self._planes) >= self.shard_size:
            self.flush()

    def add_many(self, examples) -> None:
        for planes, policy, value in examples:
            self.add(planes, policy, value)

    def flush(self) -> None:
        if not self._planes:
            return
        path = self.dir / f"{self.prefix}_{self._shard_idx:06d}.npz"
        np.savez_compressed(
            path,
            planes=np.stack(self._planes),
            policies=np.stack(self._policies),
            values=np.asarray(self._values, dtype=np.float16),
        )
        self.total_written += len(self._planes)
        self._shard_idx += 1
        self._planes.clear()
        self._policies.clear()
        self._values.clear()

    def close(self) -> None:
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def list_shards(data_dir, prefix: str = "selfplay") -> list:
    return sorted(Path(data_dir).glob(f"{prefix}_*.npz"))


def load_shard(path):
    """Return (planes, policies, values) as float32 arrays for training."""
    with np.load(path) as data:
        return (
            data["planes"].astype(np.float32),
            data["policies"].astype(np.float32),
            data["values"].astype(np.float32),
        )


def load_all(data_dir, prefix: str = "selfplay"):
    """Load every shard into three concatenated float32 arrays.

    Fine for datasets that fit in RAM; for very large corpora iterate shards
    with load_shard instead.
    """
    shards = list_shards(data_dir, prefix)
    if not shards:
        raise FileNotFoundError(f"No '{prefix}_*.npz' shards found in {data_dir}")
    planes, policies, values = [], [], []
    for path in shards:
        p, pi, v = load_shard(path)
        planes.append(p)
        policies.append(pi)
        values.append(v)
    return np.concatenate(planes), np.concatenate(policies), np.concatenate(values)


def count_examples(data_dir, prefix: str = "selfplay") -> int:
    total = 0
    for path in list_shards(data_dir, prefix):
        with np.load(path) as data:
            total += data["values"].shape[0]
    return total
