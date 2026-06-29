"""Wire framing for the inference protocol (see docs/PROTOCOL.md).

Request  frame:  count:u32 LE | planes[count*486] f32 LE
Response frame:  count:u32 LE | policy[count*209] f32 LE | value[count] f32 LE
"""

from __future__ import annotations

import socket
import struct

import numpy as np

from ..engine.game import ACTION_SIZE  # 209

PLANES = 6
BOARD = 9
PLANE_SIZE = PLANES * BOARD * BOARD  # 486
DEFAULT_SOCKET_PATH = "/tmp/quoridor_infer.sock"

_U32 = struct.Struct("<I")


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise ConnectionError on EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"socket closed with {remaining}/{n} bytes left")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_request(planes: np.ndarray) -> bytes:
    """planes: (count, 6, 9, 9) float32 (C-contiguous)."""
    count = planes.shape[0]
    arr = np.ascontiguousarray(planes, dtype="<f4")
    return _U32.pack(count) + arr.tobytes()


def read_request(sock: socket.socket) -> np.ndarray:
    """Returns (count, 6, 9, 9) float32, or raises ConnectionError on clean EOF."""
    header = recv_exact(sock, 4)
    (count,) = _U32.unpack(header)
    payload = recv_exact(sock, count * PLANE_SIZE * 4)
    return np.frombuffer(payload, dtype="<f4").reshape(count, PLANES, BOARD, BOARD)


def encode_response(policy: np.ndarray, value: np.ndarray) -> bytes:
    """policy: (count, 209) float32; value: (count,) float32."""
    count = policy.shape[0]
    p = np.ascontiguousarray(policy, dtype="<f4")
    v = np.ascontiguousarray(value, dtype="<f4")
    return _U32.pack(count) + p.tobytes() + v.tobytes()


def read_response(sock: socket.socket):
    """Returns (policy (count,209) f32, value (count,) f32)."""
    (count,) = _U32.unpack(recv_exact(sock, 4))
    pol = np.frombuffer(recv_exact(sock, count * ACTION_SIZE * 4), dtype="<f4").reshape(count, ACTION_SIZE)
    val = np.frombuffer(recv_exact(sock, count * 4), dtype="<f4").copy()
    return pol.copy(), val
