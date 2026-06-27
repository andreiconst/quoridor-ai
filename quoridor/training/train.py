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

from .network import QuoridorNet
from .selfplay import play_one_game

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def train(
    iterations: int = 50,
    games_per_iteration: int = 20,
    num_simulations: int = 100,
    buffer_size: int = 50_000,
    batch_size: int = 256,
    train_steps_per_iteration: int = 200,
    lr: float = 1e-3,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    device: str | None = None,
    resume: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    network = QuoridorNet().to(device)
    if resume:
        network.load_state_dict(torch.load(resume, map_location=device))
        print(f"Resumed from {resume}")

    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    buffer = deque(maxlen=buffer_size)

    iteration_bar = trange(1, iterations + 1, desc="Training", unit="iter")
    for it in iteration_bar:
        network.eval()
        t0 = time.time()
        results = {0: 0, 1: 0, -1: 0}
        selfplay_bar = tqdm(
            range(games_per_iteration),
            desc=f"iter {it} self-play",
            unit="game",
            leave=False,
        )
        for _ in selfplay_bar:
            examples, winner = play_one_game(network, device=device, num_simulations=num_simulations)
            buffer.extend(examples)
            results[winner] += 1
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

    return network


def main():
    parser = argparse.ArgumentParser(description="Train a Quoridor AlphaZero-style agent.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iteration", type=int, default=20)
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps-per-iteration", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        train_steps_per_iteration=args.train_steps_per_iteration,
        lr=args.lr,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
