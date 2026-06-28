"""Single-process batched self-play via asyncio.

Run K games concurrently as coroutines in one process. Each game runs a normal
(scalar) MCTS, but its leaf evaluations are `await`ed. A central batcher
collects the pending leaf requests from all active games and runs ONE network
forward, so the effective batch is ~K x (within-game leaf batch). No vectorized
MCTS, no inter-process pickling -- the batch is formed in shared memory.

Flush rule: each running game has at most one in-flight request (it awaits
before issuing another), so once `len(pending) == num_active`, every active
game is parked and we fire the largest possible batch -- no timer needed.
"""

from __future__ import annotations

import asyncio

import numpy as np
import torch

from ..engine.game import ACTION_SIZE, encode_action_for_player, encode_state, legal_action_mask
from ..engine.state import State
from .evaluator import LocalEvaluator
from .mcts import MCTS
from .network import QuoridorNet
from .selfplay import MAX_MOVES, TEMPERATURE_MOVES


class AsyncBatchEvaluator:
    """Collects awaited leaf requests across games and batches one forward."""

    def __init__(self, infer_fn):
        self.infer_fn = infer_fn  # (B,6,9,9) float32 -> (probs (B,209), values (B,))
        self._pending = []        # list of (planes_ndarray, future)
        self._num_active = 0
        self.batches = 0
        self.positions = 0

    def set_active(self, n: int) -> None:
        self._num_active = n

    def runner_done(self) -> None:
        self._num_active -= 1
        self._maybe_flush()

    async def infer(self, planes: np.ndarray):
        fut = asyncio.get_event_loop().create_future()
        self._pending.append((planes, fut))
        self._maybe_flush()
        return await fut

    def _maybe_flush(self) -> None:
        if self._pending and len(self._pending) >= self._num_active:
            self._flush()

    def _flush(self) -> None:
        items = self._pending
        self._pending = []
        planes = np.concatenate([p for p, _ in items], axis=0)
        probs, values = self.infer_fn(planes)
        self.batches += 1
        self.positions += planes.shape[0]
        offset = 0
        for p, fut in items:
            k = p.shape[0]
            if not fut.done():
                fut.set_result((probs[offset:offset + k], values[offset:offset + k]))
            offset += k


async def async_play_one_game(async_eval, num_simulations: int, mcts_batch_size: int):
    """One self-play game whose evaluations route through async_eval."""
    state = State.initial()
    mcts = MCTS(network=None, num_simulations=num_simulations, batch_size=mcts_batch_size)
    examples = []
    for ply in range(MAX_MOVES):
        if legal_action_mask(state).sum() == 0:
            break
        root = await mcts.run_async(state, async_eval, add_noise=True)
        temperature = 1.0 if ply < TEMPERATURE_MOVES else 0.1
        policy = MCTS.policy_from_visits(root, temperature=temperature)
        canonical_policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        for real_action in np.nonzero(policy)[0]:
            canonical_policy[encode_action_for_player(state, int(real_action))] = policy[real_action]
        examples.append((encode_state(state), canonical_policy, state.current_player))

        action = int(np.random.choice(len(policy), p=policy))
        state = root.children[action].state
        if state.is_terminal():
            break

    winner = state.winner
    out = []
    for planes, pol, player in examples:
        outcome = 0.0 if winner == -1 else (1.0 if player == winner else -1.0)
        out.append((planes, pol, outcome))
    return out, winner


async def _run_games(async_eval, n_games, concurrency, num_simulations, mcts_batch_size, results):
    remaining = [n_games]
    async_eval.set_active(concurrency)

    async def runner():
        while remaining[0] > 0:
            remaining[0] -= 1  # reserve a game (safe: no await between read/dec)
            ex, w = await async_play_one_game(async_eval, num_simulations, mcts_batch_size)
            results.append((ex, w))
        async_eval.runner_done()

    await asyncio.gather(*[runner() for _ in range(concurrency)])


def generate_games_async(
    state_dict,
    n_games: int,
    num_simulations: int,
    mcts_batch_size: int,
    concurrency: int,
    device: str = "cpu",
    channels: int = 64,
    blocks: int = 6,
):
    """Run n_games with `concurrency` games in flight, batched through one
    forward per step. Returns (list[(examples, winner)], stats dict)."""
    net = QuoridorNet(channels=channels, num_blocks=blocks)
    net.load_state_dict(state_dict)
    net.to(device).eval()
    evaluator = LocalEvaluator(net, device)
    async_eval = AsyncBatchEvaluator(evaluator.infer)

    results = []
    asyncio.run(_run_games(async_eval, n_games, min(concurrency, n_games),
                           num_simulations, mcts_batch_size, results))

    avg_batch = (async_eval.positions / async_eval.batches) if async_eval.batches else 0.0
    return results, {"batches": async_eval.batches, "positions": async_eval.positions,
                     "avg_batch": avg_batch}
