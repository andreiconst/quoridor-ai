# GPU Training Runbook

End-to-end recipe for training a strong Quoridor agent on a cloud GPU, encoding
the hard-won lessons from the laptop phase (see the memory/training-findings and
docs/PLAN.md). **The order matters** — the biggest time-sink on the laptop was
scaling before verifying the training loop actually compounds.

## 0. Provision + setup

Rent a single GPU box (RTX 4090 / A100 / L4 are all fine; ~$0.2–0.6/hr on
Vast.ai / RunPod). Then:

```bash
git clone <your-fork> Quorridor && cd Quorridor
bash scripts/gpu_setup.sh          # installs Go + CUDA torch, builds Go, runs tests
. .venv/bin/activate
```

`gpu_setup.sh` prints whether CUDA is visible — confirm before continuing.

## 1. Warm-start (bootstrap a skill self-play can't cold-start)

From-scratch self-play does **not** learn to reach the goal (cold-start trap).
Warm-start by imitating a heuristic. `wall_aware` (races + blocks) is the strong
one; it lifts the net from 0% → contesting a strong wall bot.

```bash
# clean value targets: random opening then DETERMINISTIC bot play to the end
python -m quoridor.training.warmstart --out-dir data/warm --games 5000 --policy wall_aware
python -m quoridor.training.pretrain  --data-dir data/warm --out checkpoints/warm.pt \
    --channels 128 --blocks 12 --epochs 15 --value-weight 2.0 --device cuda
```

Recommended net (see estimates below): **128–192 channels × 12–15 blocks (~10M params)**.

## 2. GATE: run the diagnostic BEFORE scaling

This is the single most important step. It answers "is MCTS a real improvement
operator?" — if not, no amount of self-play helps.

```bash
python scripts/diagnose.py --checkpoint checkpoints/warm.pt \
    --channels 128 --blocks 12 --device cuda --opponent wall_aware --games 60 \
    --sims 64 256 800
```

- **Green** = value calibration ≳75% **AND** win-rate **climbs** with sims. Proceed.
- **Red** = calibration weak or win-rate flat/falling with sims. Do **not** scale.
  Fix the value signal first (bigger net, higher `--value-weight`, more/better
  warm-start data, more epochs) and re-run this gate. On the laptop the value
  head capped ~68% at 3M — a larger net is the first lever.

## 3. Self-play training loop (only once the gate is green)

```bash
DEVICE=cuda CH=128 BL=12 SIMS=600 GAMES=400 CONCURRENCY=$(nproc) \
  EVAL_GAMES=60 ROUNDS=500 \
  WARM=checkpoints/warm.pt ANCHOR=data/warm \
  bash scripts/go_train_loop.sh
```

- Go actors (CPU) run parallel self-play against one batched CUDA inference
  server; the anchored learner republishes `current.pt`, hot-reloaded each round.
- Anchor-frac (0.30→0.05) and LR (3e-4→7e-5) decay across `ROUNDS` so early
  rounds protect the warm-start and late rounds let self-play move the net.
- Watch `[eval round N]`: **wall_aware should rise above 50% and hold.** Re-run
  `scripts/diagnose.py` on `checkpoints/current.pt` periodically for a
  noise-free read.

## Recommended configs (estimates — iterate!)

| Knob | Get competitive | Break ceiling / strong |
|---|---|---|
| Net size | 128ch × 12blk (~10M) | 192–256ch × 15–20blk (~15–40M) |
| Sims/move | 400–600 | 600–800 |
| Self-play games | 200k–500k | 1–5M |
| ~1× modern GPU time | ~1–3 days | ~1–4 weeks |

`games ≈ ROUNDS × GAMES`, so 500 rounds × 400 games = 200k games.

## Iteration strategy (spend cheap before spending big)

1. Short runs (20–50k games) tuning **value quality** (net size, `--value-weight`,
   warm-start data), judged by `scripts/diagnose.py` — not by a full run.
2. Only pour a big game budget into a config whose diagnostic is green.
3. Always eval with **randomized openings and ≥50 games** — deterministic /
   small-N evals are misleading (they showed fake 100%s on the laptop).
4. Track win-rate vs `wall_aware` (absolute) and self-gating vs earlier
   checkpoints (relative, non-saturating).

## Persisting results

Cloud disks are often ephemeral — `rsync`/`scp` `checkpoints/` and `data/` off
the box periodically, or write them to a mounted volume. `current.pt` is written
atomically each round, so it's always safe to copy.
