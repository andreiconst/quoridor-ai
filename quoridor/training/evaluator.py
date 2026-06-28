"""Inference evaluators for MCTS, including a centralized batched server.

Two ways MCTS gets policy/value predictions:

- LocalEvaluator: runs the network in-process (default; one net per worker).
- RemoteEvaluator + evaluation_server: many CPU self-play *actors* send leaf
  positions to a single *server* process that batches across ALL actors into
  one big forward pass (ideal for a GPU). This is the KataGo/Leela pattern.

The server forms batches from whatever requests have arrived within a short
window, so the effective batch grows with the number of in-flight actors
(times each actor's within-search virtual-loss batch).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty

import numpy as np
import torch

from .network import QuoridorNet


class LocalEvaluator:
    """In-process network inference."""

    def __init__(self, network, device: str = "cpu"):
        self.network = network
        self.device = device

    def infer(self, planes: np.ndarray):
        """planes: (B, 6, 9, 9) float32 -> (probs (B,209), values (B,))."""
        tensor = torch.from_numpy(planes).to(self.device)
        probs, values = self.network.predict(tensor)
        return probs.cpu().numpy(), values.cpu().numpy()


class RemoteEvaluator:
    """Client side: round-trips a leaf batch to the central server."""

    def __init__(self, actor_id: int, request_q, response_q):
        self.actor_id = actor_id
        self.request_q = request_q
        self.response_q = response_q

    def infer(self, planes: np.ndarray):
        self.request_q.put((self.actor_id, planes))
        return self.response_q.get()


def evaluation_server(state_dict, request_q, response_qs, stop_event, stats,
                      device="cpu", channels=64, blocks=6,
                      max_batch=256, max_wait_ms=2.0, server_threads=0):
    """Collect requests from all actors, batch, run one forward, scatter back."""
    if server_threads:
        torch.set_num_threads(server_threads)
    net = QuoridorNet(channels=channels, num_blocks=blocks)
    net.load_state_dict(state_dict)
    net.to(device).eval()
    evaluator = LocalEvaluator(net, device)
    max_wait = max_wait_ms / 1000.0

    while True:
        try:
            first = request_q.get(timeout=0.1)
        except Empty:
            if stop_event.is_set():
                break
            continue

        items = [first]
        total = first[1].shape[0]
        deadline = time.time() + max_wait
        # Greedily absorb whatever else is already queued, up to max_batch.
        while total < max_batch:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                nxt = request_q.get(timeout=remaining)
            except Empty:
                break
            items.append(nxt)
            total += nxt[1].shape[0]

        planes = np.concatenate([p for _, p in items], axis=0)
        probs, values = evaluator.infer(planes)

        offset = 0
        for actor_id, p in items:
            k = p.shape[0]
            response_qs[actor_id].put((probs[offset:offset + k], values[offset:offset + k]))
            offset += k

        # Running stats: number of forward passes and total positions, so the
        # caller can report the average batch size actually achieved.
        with stats["lock"]:
            stats["batches"].value += 1
            stats["positions"].value += total


def _actor_loop(actor_id, request_q, response_q, game_specs, result_q):
    torch.set_num_threads(1)  # actors are CPU tree-walk only
    from .selfplay import play_one_game

    evaluator = RemoteEvaluator(actor_id, request_q, response_q)
    for num_simulations, mcts_batch_size in game_specs:
        examples, winner = play_one_game(
            evaluator=evaluator,
            num_simulations=num_simulations,
            mcts_batch_size=mcts_batch_size,
        )
        result_q.put((examples, winner))


def generate_games_server(
    state_dict,
    n_games: int,
    num_simulations: int,
    mcts_batch_size: int,
    n_actors: int,
    device: str = "cpu",
    channels: int = 64,
    blocks: int = 6,
    max_batch: int = 256,
    max_wait_ms: float = 2.0,
    server_threads: int = 0,
):
    """Run n_games via n_actors CPU actors feeding one batched server.

    Yields (examples, winner) as games complete. Also returns final batch
    stats through the generator's `.stats` attribute is awkward, so we attach
    them to the returned dict via the `info` mutable passed back at the end:
    instead we just print/return them from the helper below.
    """
    ctx = mp.get_context("spawn")
    request_q = ctx.Queue()
    response_qs = [ctx.Queue() for _ in range(n_actors)]
    result_q = ctx.Queue()
    stop_event = ctx.Event()
    stats = {
        "batches": ctx.Value("l", 0),
        "positions": ctx.Value("l", 0),
        "lock": ctx.Lock(),
    }

    cpu_state = {k: v.cpu() for k, v in state_dict.items()}

    server = ctx.Process(
        target=evaluation_server,
        args=(cpu_state, request_q, response_qs, stop_event, stats),
        kwargs=dict(device=device, channels=channels, blocks=blocks,
                    max_batch=max_batch, max_wait_ms=max_wait_ms,
                    server_threads=server_threads),
    )
    server.start()

    # Distribute games across actors.
    counts = [n_games // n_actors] * n_actors
    for i in range(n_games % n_actors):
        counts[i] += 1

    actors = []
    for aid in range(n_actors):
        specs = [(num_simulations, mcts_batch_size)] * counts[aid]
        p = ctx.Process(
            target=_actor_loop,
            args=(aid, request_q, response_qs[aid], specs, result_q),
        )
        p.start()
        actors.append(p)

    collected = []
    for _ in range(n_games):
        collected.append(result_q.get())

    for p in actors:
        p.join()
    stop_event.set()
    server.join()

    n_batches = stats["batches"].value
    n_positions = stats["positions"].value
    avg_batch = (n_positions / n_batches) if n_batches else 0.0
    return collected, {"batches": n_batches, "positions": n_positions, "avg_batch": avg_batch}
