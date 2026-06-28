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
from tqdm import tqdm, trange

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
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    device: str | None = None,
    resume: str | None = None,
):
    device = device or _auto_device()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    network = QuoridorNet().to(device)
    if resume:
        network.load_state_dict(torch.load(resume, map_location=device))
        print(f"Resumed from {resume}")

    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    buffer = deque(maxlen=buffer_size)
    best_win_rate = -1.0

    iteration_bar = trange(1, iterations + 1, desc="Training", unit="iter")
    for it in iteration_bar:
        network.eval()
        t0 = time.time()
        results = {0: 0, 1: 0, -1: 0}
        selfplay_bar = tqdm(
            total=games_per_iteration,
            desc=f"iter {it} self-play",
            unit="game",
            leave=False,
        )
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
            results[winner] += 1
            selfplay_bar.update(1)
            selfplay_bar.set_postfix(P0=results[0], P1=results[1], draw=results[-1], buffer=len(buffer))
        selfplay_bar.close()
        gen_time = time.time() - t0

        network.train()
        losses = []
        if len(buffer) >= batch_size:
            train_bar = tqdm(
                range(train_steps_per_iteration),
                desc=f"iter {it} train",
                unit="step",
                leave=False,
            )
            for _ in train_bar:
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
                train_bar.set_postfix(loss=f"{loss.item():.4f}")
            train_bar.close()

        avg_loss = np.mean(losses) if losses else float("nan")
        # tqdm.write keeps the per-iteration summary from clobbering the bars.
        tqdm.write(
            f"[iter {it}/{iterations}] games: P0={results[0]} P1={results[1]} draw={results[-1]} "
            f"| buffer={len(buffer)} | loss={avg_loss:.4f} | selfplay_time={gen_time:.1f}s"
        )
        iteration_bar.set_postfix(loss=f"{avg_loss:.4f}", buffer=len(buffer))

        ckpt_path = checkpoint_dir / f"model_iter{it}.pt"
        torch.save(network.state_dict(), ckpt_path)
        torch.save(network.state_dict(), checkpoint_dir / "latest.pt")

        # Periodic strength check against the shortest-path baseline.
        if eval_interval and (it % eval_interval == 0 or it == iterations):
            network.eval()
            wins, draws, losses_ = evaluate_vs_baseline(
                network, n_games=eval_games, num_simulations=num_simulations,
                device=device, mcts_batch_size=mcts_batch_size,
            )
            win_rate = wins / max(eval_games, 1)
            tqdm.write(
                f"    [eval iter {it}] vs shortest-path baseline: "
                f"{wins}W-{draws}D-{losses_}L  win_rate={win_rate:.0%}"
            )
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(network.state_dict(), checkpoint_dir / "best.pt")
                tqdm.write(f"    new best win_rate={win_rate:.0%} -> saved best.pt")
            iteration_bar.set_postfix(loss=f"{avg_loss:.4f}", win_rate=f"{win_rate:.0%}")

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
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
