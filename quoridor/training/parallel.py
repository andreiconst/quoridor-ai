"""Parallel self-play across worker processes.

Self-play is CPU-bound pure-Python (MCTS tree walk + wall-legality), so the GIL
serializes it within one process. Separate processes sidestep the GIL, and
games are fully independent, so this scales with physical core count (minus
heterogeneous-core / memory-bandwidth effects).

Each worker pins itself to a single torch thread -- otherwise N worker
processes each spawn N BLAS threads and thrash, erasing the parallelism gain.
The network is rebuilt once per worker (via the pool initializer) rather than
per game.
"""

from __future__ import annotations

import multiprocessing as mp

import torch

_WORKER = {}


def _init_worker(state_dict, num_simulations: int, mcts_batch_size: int) -> None:
    torch.set_num_threads(1)
    from .network import QuoridorNet

    net = QuoridorNet()
    net.load_state_dict(state_dict)
    net.eval()
    _WORKER["net"] = net
    _WORKER["sims"] = num_simulations
    _WORKER["bs"] = mcts_batch_size


def _play_task(_) -> tuple:
    from .selfplay import play_one_game

    return play_one_game(
        _WORKER["net"],
        device="cpu",
        num_simulations=_WORKER["sims"],
        mcts_batch_size=_WORKER["bs"],
    )


def generate_games_parallel(
    state_dict,
    n_games: int,
    num_simulations: int,
    mcts_batch_size: int,
    workers: int,
):
    """Yield (examples, winner) tuples as games finish, using `workers`
    processes. Workers always run on CPU (fast for the small net, and avoids
    sharing a GPU context across processes)."""
    cpu_state = {k: v.cpu() for k, v in state_dict.items()}
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=(cpu_state, num_simulations, mcts_batch_size),
    ) as pool:
        for result in pool.imap_unordered(_play_task, range(n_games)):
            yield result
