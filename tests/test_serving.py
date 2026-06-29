"""Phase 1 end-to-end: the socket inference server must return exactly what a
direct in-process forward returns, and must drive a full MCTS self-play game."""

import multiprocessing as mp
import os
import time
import uuid

import numpy as np
import pytest
import torch

from quoridor.engine.game import encode_state
from quoridor.engine.state import State
from quoridor.training.network import QuoridorNet
from quoridor.training.mcts import MCTS
from quoridor.serving import protocol
from quoridor.serving.client import InferenceClient
from quoridor.serving.infer_server import InferenceModel, serve


def _server_proc(socket_path, ckpt):
    model = InferenceModel(ckpt, device="cpu", channels=16, blocks=2)
    serve(socket_path, model)


@pytest.fixture()
def running_server(tmp_path):
    ckpt = tmp_path / "net.pt"
    torch.manual_seed(0)
    net = QuoridorNet(channels=16, num_blocks=2)
    torch.save(net.state_dict(), ckpt)
    # AF_UNIX paths are limited (~104 chars on macOS); keep it short + unique
    # (terminate() skips the server's cleanup, so avoid stale-path reuse).
    sock = f"/tmp/q_{uuid.uuid4().hex[:8]}.sock"
    if os.path.exists(sock):
        os.unlink(sock)

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_server_proc, args=(sock, str(ckpt)), daemon=True)
    proc.start()
    for _ in range(200):  # wait until the server is actually accepting
        if os.path.exists(sock):
            try:
                import socket as _s
                c = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM); c.connect(sock); c.close()
                break
            except OSError:
                pass
        time.sleep(0.05)
    yield sock, ckpt, net
    proc.terminate()
    proc.join(timeout=5)
    if os.path.exists(sock):
        os.unlink(sock)


def test_server_matches_local_forward(running_server):
    sock, _ckpt, net = running_server
    net.eval()

    states = [State.initial()]
    s = State.initial(); s.pawns = [(3, 4), (4, 4)]; s.current_player = 1
    states.append(s)
    planes = np.stack([encode_state(st) for st in states]).astype(np.float32)

    with InferenceClient(sock) as client:
        probs, values = client.infer(planes)

    ref_probs, ref_values = net.predict(torch.from_numpy(planes))
    assert np.allclose(probs, ref_probs.numpy(), atol=1e-5)
    assert np.allclose(values, ref_values.numpy(), atol=1e-5)


def test_client_drives_mcts_game(running_server):
    sock, _ckpt, _net = running_server
    with InferenceClient(sock) as client:
        mcts = MCTS(evaluator=client, num_simulations=20, batch_size=8)
        state = State.initial()
        for _ in range(6):
            root = mcts.run(state, add_noise=False)
            action = int(MCTS.policy_from_visits(root, temperature=0.0).argmax())
            state = root.children[action].state
            if state.is_terminal():
                break
        assert state is not None  # completed without protocol errors
