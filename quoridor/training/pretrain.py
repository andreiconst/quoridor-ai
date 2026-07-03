"""Supervised warm-start of a network from saved self-play games.

When you change the architecture, you don't have to start from scratch: the
stored (planes, policy, value) tuples are labelled data. Training a fresh
network to imitate those MCTS policies and game outcomes recovers most of the
strength immediately, and you can then continue self-play with --resume.

  python -m quoridor.training.pretrain --data-dir data/ --epochs 10 \
      --channels 96 --blocks 10 --out checkpoints/pretrained.pt
  python -m quoridor.training.train --resume checkpoints/pretrained.pt \
      --channels 96 --blocks 10 ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from .dataset import count_examples, load_all
from .network import QuoridorNet


def pretrain(
    data_dir: str,
    out: str,
    channels: int = 64,
    blocks: int = 6,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_frac: float = 0.05,
    value_weight: float = 1.0,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading dataset from {data_dir} ({count_examples(data_dir):,} examples)...")
    planes, policies, values = load_all(data_dir)
    n = planes.shape[0]

    # Shuffle and split off a small validation set.
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    planes, policies, values = planes[perm], policies[perm], values[perm]
    n_val = int(n * val_frac)
    tr = slice(n_val, n)
    va = slice(0, n_val)

    planes_t = torch.from_numpy(planes)
    policies_t = torch.from_numpy(policies)
    values_t = torch.from_numpy(values)

    net = QuoridorNet(channels=channels, num_blocks=blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    def loss_on(idx):
        x = planes_t[idx].to(device)
        tp = policies_t[idx].to(device)
        tv = values_t[idx].to(device)
        logits, value = net(x)
        policy_loss = -(tp * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, tv)
        return policy_loss, value_loss

    train_idx = np.arange(n_val, n)
    for epoch in range(1, epochs + 1):
        net.train()
        rng.shuffle(train_idx)
        bar = trange(0, len(train_idx), batch_size, desc=f"epoch {epoch}/{epochs}", leave=False)
        running = []
        for start in bar:
            idx = train_idx[start:start + batch_size]
            pl, vl = loss_on(idx)
            loss = pl + value_weight * vl
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        net.eval()
        with torch.no_grad():
            if n_val > 0:
                vpl, vvl = loss_on(np.arange(0, n_val))
                vmsg = f"val_policy={vpl.item():.4f} val_value={vvl.item():.4f}"
            else:
                vmsg = "(no val split)"
        print(f"[epoch {epoch}/{epochs}] train_loss={np.mean(running):.4f} | {vmsg}")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), out)

    print(f"Saved warm-started network to {out}")
    return net


def main():
    p = argparse.ArgumentParser(description="Supervised pretraining from saved self-play games.")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out", type=str, default="checkpoints/pretrained.pt")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    pretrain(
        data_dir=args.data_dir, out=args.out, channels=args.channels, blocks=args.blocks,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_frac=args.val_frac, value_weight=args.value_weight, device=args.device,
    )


if __name__ == "__main__":
    main()
