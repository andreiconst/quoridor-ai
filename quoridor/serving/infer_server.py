"""Phase 1: Python inference server over a Unix domain socket.

Serves the request/response frames in docs/PROTOCOL.md: receives a batch of
encoded planes, runs one QuoridorNet forward, returns policy probabilities and
values. Hot-reloads weights when the checkpoint file changes.

This is the standalone inference role of the Go/Python split. For now it's
exercised by a Python client (quoridor.serving.client); later the Go self-play
batcher speaks the same protocol.

    python -m quoridor.serving.infer_server --checkpoint checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

import numpy as np
import torch

from ..training.network import QuoridorNet
from . import protocol


class InferenceModel:
    """Holds the network and hot-reloads it when the checkpoint changes."""

    def __init__(self, checkpoint: str | None, device: str, channels: int, blocks: int):
        self.device = device
        self.channels = channels
        self.blocks = blocks
        self.checkpoint = checkpoint
        self._mtime = None
        self.net = QuoridorNet(channels=channels, num_blocks=blocks).to(device).eval()
        self._maybe_reload(force=True)

    def _maybe_reload(self, force: bool = False) -> None:
        if not self.checkpoint or not os.path.exists(self.checkpoint):
            return
        mtime = os.path.getmtime(self.checkpoint)
        if force or mtime != self._mtime:
            self.net.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
            self.net.eval()
            self._mtime = mtime
            print(f"[infer] loaded weights from {self.checkpoint} (mtime={mtime:.0f})", flush=True)

    @torch.no_grad()
    def infer(self, planes: np.ndarray):
        self._maybe_reload()
        # copy=True yields a writable array (the wire buffer is read-only).
        tensor = torch.from_numpy(np.array(planes, dtype=np.float32)).to(self.device)
        probs, values = self.net.predict(tensor)
        return probs.cpu().numpy(), values.cpu().numpy()


def serve(socket_path: str, model: InferenceModel) -> None:
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(16)
    print(f"[infer] listening on {socket_path} (device={model.device})", flush=True)
    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    while True:
                        planes = protocol.read_request(conn)  # raises on clean EOF
                        probs, values = model.infer(planes)
                        conn.sendall(protocol.encode_response(probs, values))
                except (ConnectionError, OSError):
                    pass  # client disconnected; wait for the next one
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


def main():
    p = argparse.ArgumentParser(description="Quoridor inference server (Unix socket).")
    p.add_argument("--socket", default=protocol.DEFAULT_SOCKET_PATH)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    args = p.parse_args()
    model = InferenceModel(args.checkpoint, args.device, args.channels, args.blocks)
    serve(args.socket, model)


if __name__ == "__main__":
    main()
