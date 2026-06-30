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
from .evaluate import evaluate_vs_baseline
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
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    device: str | None = None,
    resume: str | None = None,
):
    device = device or _auto_device()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(data_dir) if data_dir else None

    network = QuoridorNet(channels=channels, num_blocks=blocks).to(device)
    if resume:
        network.load_state_dict(torch.load(resume, map_location=device))
        print(f"Resumed from {resume}")

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
            )
        else:
            game_results = (
                play_one_game(network, device=device, num_simulations=num_simulations,
                              mcts_batch_size=mcts_batch_size)
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
        if len(buffer) >= batch_size:
            for _ in range(train_steps_per_iteration):
                batch = random.sample(buffer, batch_size)
                planes = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
                target_policy = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
                target_value = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)

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

        # Periodic strength check: vs random (easy, shows early progress) and
        # vs the shortest-path baseline (hard, real strength).
        if eval_interval and (it % eval_interval == 0 or it == iterations):
            network.eval()
            rw, rd, rl = evaluate_vs_baseline(
                network, n_games=eval_games, num_simulations=num_simulations,
                device=device, mcts_batch_size=mcts_batch_size, opponent="random",
            )
            sw, sd, sl = evaluate_vs_baseline(
                network, n_games=eval_games, num_simulations=num_simulations,
                device=device, mcts_batch_size=mcts_batch_size, opponent="shortest_path",
            )
            rand_wr = rw / max(eval_games, 1)
            sp_wr = sw / max(eval_games, 1)
            msg = (
                f"    [eval iter {it}] vs random: {rand_wr:.0%} ({rw}W-{rd}D-{rl}L)  "
                f"| vs shortest-path: {sp_wr:.0%} ({sw}W-{sd}D-{sl}L)"
            )
            if sp_wr > best_win_rate:
                best_win_rate = sp_wr
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
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
