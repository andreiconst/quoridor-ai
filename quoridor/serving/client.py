"""Phase 1: Python client for the inference server.

Implements the same wire framing as a future Go client, and exposes the
`infer(planes) -> (probs, values)` interface used by MCTS evaluators, so it can
drop in as a RemoteEvaluator-over-socket and validate the protocol end-to-end.
"""

from __future__ import annotations

import socket

import numpy as np

from . import protocol


class InferenceClient:
    def __init__(self, socket_path: str = protocol.DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(socket_path)

    def infer(self, planes: np.ndarray):
        """planes: (count, 6, 9, 9) float32 -> (probs (count,209), values (count,))."""
        self.sock.sendall(protocol.encode_request(planes))
        return protocol.read_response(self.sock)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
