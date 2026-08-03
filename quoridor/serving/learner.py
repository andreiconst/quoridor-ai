"""Learner: train QuoridorNet from Go self-play shards and publish weights.

Reads (planes, policy, value) examples produced by the Go self-play process,
runs supervised AlphaZero-style updates (policy cross-entropy + value MSE), and
writes the checkpoint *atomically* (tmp + os.replace) so the inference server
can hot-reload it mid-run without ever reading a partial file.

    python -m quoridor.serving.learner --data-dir data/ --out checkpoints/current.pt \
        --steps 500 --resume checkpoints/current.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..training.network import QuoridorNet
from .go_shards import load_go_dir


def atomic_save(state_dict, out: str) -> None:
    tmp = out + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, out)  # atomic on POSIX; readers never see a partial file


def train_once(
    data_dir: str,
    out: str,
    channels: int = 64,
    blocks: int = 6,
    steps: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    resume: str | None = None,
    device: str | None = None,
    anchor_data: str | None = None,
    anchor_frac: float = 0.25,
    outcome_weight: float = 0.6,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    planes, policies, values, search_values = load_go_dir(data_dir)
    n = len(values)
    print(f"[learner] {n} examples from {data_dir}", flush=True)

    # Anchor set (e.g. warm-start data) mixed into every batch to prevent
    # catastrophic forgetting during self-play fine-tuning.
    a_planes = a_policies = a_values = None
    n_anchor = 0
    if anchor_data:
        from ..training.dataset import load_all
        a_planes, a_policies, a_values = load_all(anchor_data)
        n_anchor = int(round(batch_size * anchor_frac))
        print(f"[learner] anchoring on {a_values.shape[0]} examples ({n_anchor}/{batch_size} per batch)", flush=True)

    net = QuoridorNet(channels=channels, num_blocks=blocks).to(device)
    if resume and os.path.exists(resume):
        net.load_state_dict(torch.load(resume, map_location=device))
        print(f"[learner] resumed {resume}", flush=True)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    planes_t = torch.from_numpy(planes)
    policy_t = torch.from_numpy(policies)
    value_t = torch.from_numpy(values)
    search_t = torch.from_numpy(search_values)

    losses = []
    n_self = batch_size - n_anchor
    for _ in range(steps):
        idx = np.random.randint(0, n, size=min(n_self, n))
        ns = len(idx)  # actual self-play rows in this batch (may be < n_self early on)
        bx, bp = planes_t[idx], policy_t[idx]
        # Blended value target (lever 2): the final outcome anchors it to ground
        # truth while the lower-variance MCTS search value sharpens calibration,
        # so the value scales with search instead of misleading it at high sims.
        bv = outcome_weight * value_t[idx] + (1.0 - outcome_weight) * search_t[idx]
        if n_anchor > 0:
            ai = np.random.randint(0, a_planes.shape[0], size=n_anchor)
            bx = torch.cat([bx, torch.from_numpy(a_planes[ai])])
            bp = torch.cat([bp, torch.from_numpy(a_policies[ai])])
        x = bx.to(device)
        tp = bp.to(device)
        tv = bv.to(device)
        logits, value = net(x)
        # Policy loss on the FULL batch (anchor protects racing skill from being
        # forgotten). Value loss on the SELF-PLAY rows ONLY (lever 1): the anchor's
        # value labels are the wall_aware heuristic's playout outcomes, and training
        # the value head on them pins it to the heuristic's ~63% ceiling.
        policy_loss = -(tp * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        value_loss = F.mse_loss(value[:ns], tv)
        loss = policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    atomic_save(net.state_dict(), out)
    print(f"[learner] trained {steps} steps, mean_loss={np.mean(losses):.4f}, published {out}", flush=True)
    return net


def main():
    p = argparse.ArgumentParser(description="Train from Go self-play shards; publish weights.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="checkpoints/current.pt")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--anchor-data", default=None)
    p.add_argument("--anchor-frac", type=float, default=0.25)
    p.add_argument("--outcome-weight", type=float, default=0.6,
                   help="value target = w*outcome + (1-w)*mcts_search_value")
    args = p.parse_args()
    train_once(
        data_dir=args.data_dir, out=args.out, channels=args.channels, blocks=args.blocks,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, resume=args.resume, device=args.device,
        anchor_data=args.anchor_data, anchor_frac=args.anchor_frac, outcome_weight=args.outcome_weight,
    )


if __name__ == "__main__":
    main()
