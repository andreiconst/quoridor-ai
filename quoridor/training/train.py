"""AlphaZero-style training loop: self-play -> replay buffer -> SGD -> repeat."""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import ShardWriter
from .evaluate import evaluate_vs_baseline, evaluate_vs_net
from .network import QuoridorNet
from .parallel import generate_games_parallel
from .selfplay import play_one_game

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def _auto_device() -> str:
    # Note: we deliberately do NOT auto-select Apple MPS. For the default small
    # network, per-call host<->GPU transfer overhead during self-play makes MPS
    # several times slower than CPU end-to-end. MPS/CUDA only pay off once the
    # network is large enough and the MCTS batch size high enough that compute
    # dominates transfer -- pass --device mps (or cuda) explicitly then.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(
    iterations: int = 50,
    games_per_iteration: int = 20,
    num_simulations: int = 100,
    buffer_size: int = 50_000,
    batch_size: int = 256,
    train_steps_per_iteration: int = 200,
    lr: float = 1e-3,
    mcts_batch_size: int = 16,
    workers: int = 1,
    eval_interval: int = 5,
    eval_games: int = 20,
    channels: int = 64,
    blocks: int = 6,
    data_dir: str | None = None,
    gate_vs: str | None = None,
    anchor_data: str | None = None,
    anchor_frac: float = 0.25,
    opponent_prob: float = 0.0,
    gate_promote: float = 0.6,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    device: str | None = None,
    resume: str | None = None,
):
    device = device or _auto_device()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(data_dir) if data_dir else None

    # Anchor set (e.g. warm-start racing data) mixed into every batch to prevent
    # catastrophic forgetting during self-play fine-tuning.
    anchor_planes = anchor_policies = anchor_values = None
    n_anchor = 0
    if anchor_data:
        from .dataset import load_all
        anchor_planes, anchor_policies, anchor_values = load_all(anchor_data)
        n_anchor = int(round(batch_size * anchor_frac))
        print(f"Anchoring on {anchor_values.shape[0]} examples from {anchor_data} "
              f"({n_anchor}/{batch_size} per batch)")

    network = QuoridorNet(channels=channels, num_blocks=blocks).to(device)
    if resume:
        network.load_state_dict(torch.load(resume, map_location=device))
        print(f"Resumed from {resume}")

    # Frozen reference net for non-saturating self-gating eval (e.g. warm-start).
    reference_net = None
    if gate_vs:
        reference_net = QuoridorNet(channels=channels, num_blocks=blocks).to(device)
        reference_net.load_state_dict(torch.load(gate_vs, map_location=device))
        reference_net.eval()
        print(f"Gating eval vs frozen reference {gate_vs}")

    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    buffer = deque(maxlen=buffer_size)
    best_win_rate = -1.0

    for it in range(1, iterations + 1):
        network.eval()
        t0 = time.time()
        results = {0: 0, 1: 0, -1: 0}
        if workers and workers > 1:
            # Parallel self-play on CPU worker processes.
            game_results = generate_games_parallel(
                network.state_dict(), games_per_iteration,
                num_simulations, mcts_batch_size, workers,
                channels=channels, blocks=blocks, opponent_prob=opponent_prob,
            )
        else:
            game_results = (
                play_one_game(network, device=device, num_simulations=num_simulations,
                              mcts_batch_size=mcts_batch_size, opponent_prob=opponent_prob)
                for _ in range(games_per_iteration)
            )
        for examples, winner in game_results:
            buffer.extend(examples)
            if writer is not None:
                writer.add_many(examples)
            results[winner] += 1
        gen_time = time.time() - t0

        network.train()
        losses = []
        n_self = batch_size - n_anchor
        if len(buffer) >= n_self:
            for _ in range(train_steps_per_iteration):
                batch = random.sample(buffer, n_self)
                planes_np = np.stack([b[0] for b in batch])
                policy_np = np.stack([b[1] for b in batch])
                value_np = np.array([b[2] for b in batch], dtype=np.float32)
                if n_anchor > 0:
                    # Mix in warm-start examples so racing isn't forgotten while
                    # self-play teaches walls (anchored fine-tuning).
                    idx = np.random.randint(0, anchor_planes.shape[0], size=n_anchor)
                    planes_np = np.concatenate([planes_np, anchor_planes[idx]])
                    policy_np = np.concatenate([policy_np, anchor_policies[idx]])
                    value_np = np.concatenate([value_np, anchor_values[idx]])
                planes = torch.from_numpy(planes_np).to(device)
                target_policy = torch.from_numpy(policy_np).to(device)
                target_value = torch.from_numpy(value_np).to(device)

                logits, value = network(planes)
                policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                value_loss = F.mse_loss(value, target_value)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        avg_loss = np.mean(losses) if losses else float("nan")
        train_time = time.time() - t0 - gen_time
        # One concise summary line per iteration.
        print(
            f"[iter {it}/{iterations}] games: P0={results[0]} P1={results[1]} draw={results[-1]} "
            f"| buffer={len(buffer)} | loss={avg_loss:.4f} "
            f"| selfplay={gen_time:.1f}s train={train_time:.1f}s",
            flush=True,
        )

        ckpt_path = checkpoint_dir / f"model_iter{it}.pt"
        torch.save(network.state_dict(), ckpt_path)
        torch.save(network.state_dict(), checkpoint_dir / "latest.pt")
        if writer is not None:
            writer.flush()  # persist this iteration's games (crash-safe on spot VMs)

        # Periodic strength check. shortest-path saturates once the net can
        # race, so wall_aware is the absolute yardstick and the frozen reference
        # net (gate) is the non-saturating "better than my past self" signal.
        if eval_interval and (it % eval_interval == 0 or it == iterations):
            network.eval()

            def _wr(opponent):
                w, d, l = evaluate_vs_baseline(
                    network, n_games=eval_games, num_simulations=num_simulations,
                    device=device, mcts_batch_size=mcts_batch_size, opponent=opponent,
                )
                return w / max(eval_games, 1), f"{w}W-{d}D-{l}L"

            sp_wr, sp_s = _wr("shortest_path")
            wa_wr, wa_s = _wr("wall_aware")
            msg = (f"    [eval iter {it}] vs shortest-path: {sp_wr:.0%} ({sp_s})  "
                   f"| vs wall-aware: {wa_wr:.0%} ({wa_s})")

            if reference_net is not None:
                gw, gd, gl = evaluate_vs_net(
                    network, reference_net, n_games=eval_games,
                    num_simulations=num_simulations, device=device,
                    mcts_batch_size=mcts_batch_size,
                )
                gate_wr = gw / max(eval_games, 1)
                msg += f"  | vs gate: {gate_wr:.0%} ({gw}W-{gd}D-{gl}L)"
                # Advancing gate (Elo ladder): once the net clearly beats the
                # reference, promote the reference to the current net so the
                # gate keeps measuring progress instead of saturating.
                if gate_wr >= gate_promote:
                    reference_net.load_state_dict(network.state_dict())
                    torch.save(network.state_dict(), checkpoint_dir / "gate.pt")
                    msg += " [gate promoted]"

            if wa_wr > best_win_rate:
                best_win_rate = wa_wr
                torch.save(network.state_dict(), checkpoint_dir / "best.pt")
                msg += "  -> new best, saved best.pt"
            print(msg, flush=True)

    if writer is not None:
        writer.close()
    return network


def main():
    parser = argparse.ArgumentParser(description="Train a Quoridor AlphaZero-style agent.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iteration", type=int, default=20)
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256, help="SGD training batch size")
    parser.add_argument("--train-steps-per-iteration", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mcts-batch-size", type=int, default=16,
                        help="MCTS leaf-evaluation batch size (helps most on GPU/MPS)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel self-play worker processes (CPU). 1 = serial.")
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Evaluate vs baseline every N iterations (0 to disable)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Number of arena games per evaluation")
    parser.add_argument("--channels", type=int, default=64, help="Conv channels (network width)")
    parser.add_argument("--blocks", type=int, default=6, help="Residual blocks (network depth)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="If set, save all self-play (state, policy, outcome) examples here")
    parser.add_argument("--gate-vs", type=str, default=None,
                        help="Frozen reference checkpoint for self-gating eval (e.g. the warm-start net)")
    parser.add_argument("--anchor-data", type=str, default=None,
                        help="Mix this dataset (e.g. warm-start shards) into every batch to prevent forgetting")
    parser.add_argument("--anchor-frac", type=float, default=0.25,
                        help="Fraction of each training batch drawn from the anchor set")
    parser.add_argument("--opponent-prob", type=float, default=0.0,
                        help="Fraction of self-play games vs a heuristic bot (opponent diversity)")
    parser.add_argument("--gate-promote", type=float, default=0.6,
                        help="Promote the gate reference once the net's win-rate vs it reaches this")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps (auto if unset)")
    args = parser.parse_args()

    train(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        train_steps_per_iteration=args.train_steps_per_iteration,
        lr=args.lr,
        mcts_batch_size=args.mcts_batch_size,
        workers=args.workers,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        channels=args.channels,
        blocks=args.blocks,
        data_dir=args.data_dir,
        gate_vs=args.gate_vs,
        anchor_data=args.anchor_data,
        anchor_frac=args.anchor_frac,
        opponent_prob=args.opponent_prob,
        gate_promote=args.gate_promote,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
