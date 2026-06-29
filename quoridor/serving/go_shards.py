"""Reader for the Go self-play shard format (see go/data/shard.go).

Format per .qsh file (little-endian):
    magic u32 (0x51534831) | count u32 | planeSize u32 (486) | actionSize u32 (209)
    then count records: planes[486] f32 | policy[209] f32 | value f32
"""

from __future__ import annotations

import glob
import os
import struct

import numpy as np

MAGIC = 0x51534831
PLANE_SIZE = 486
ACTION_SIZE = 209


def load_go_shard(path):
    """Returns (planes (N,6,9,9), policies (N,209), values (N,)) float32."""
    with open(path, "rb") as f:
        raw = f.read()
    magic, count, ps, asz = struct.unpack_from("<IIII", raw, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x} in {path}")
    if ps != PLANE_SIZE or asz != ACTION_SIZE:
        raise ValueError(f"unexpected sizes ps={ps} asz={asz} in {path}")
    rec = ps + asz + 1
    arr = np.frombuffer(raw, dtype="<f4", offset=16, count=count * rec).reshape(count, rec)
    planes = arr[:, :ps].reshape(count, 6, 9, 9).astype(np.float32)
    policies = arr[:, ps:ps + asz].astype(np.float32)
    values = arr[:, ps + asz].astype(np.float32)
    return planes, policies, values


def list_go_shards(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "go_*.qsh")))


def load_go_dir(data_dir):
    """Concatenate every Go shard in a directory."""
    shards = list_go_shards(data_dir)
    if not shards:
        raise FileNotFoundError(f"no go_*.qsh shards in {data_dir}")
    planes, policies, values = [], [], []
    for path in shards:
        p, po, v = load_go_shard(path)
        planes.append(p)
        policies.append(po)
        values.append(v)
    return np.concatenate(planes), np.concatenate(policies), np.concatenate(values)
