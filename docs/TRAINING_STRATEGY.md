# Training Strategy: getting to competitive strength

A phased plan to train a strong Quoridor agent on a single cloud GPU, plus the
exact commands to launch it crash-safely. This complements
[GPU_RUNBOOK.md](GPU_RUNBOOK.md) (the recipe) and encodes the lessons in
`memory/training-findings` and [PLAN.md](PLAN.md).

## The one idea that reframes everything: train-sims ≠ play-sims

Competitive Quoridor engines search **10k–100k MCTS simulations per move** — but
that is a *deployment / tournament* number, not a training one. AlphaZero
generated its self-play games at **800 sims/move** and became superhuman. The
network is the durable asset; simulations are a strength amplifier you dial up
**for free at play time**.

So we **never** train self-play at 100k sims (slower data, and — proven on this
project — unstable). We train at modest sims, then crank sims at deploy time.

### The precondition that gates the whole plan

Extra search only helps if strength **climbs monotonically with sims**. This
project already hit the failure mode where it does *not*: a miscalibrated value
head made `warm3m` play **worse** at 800/1600 sims than at 64 (`25% → 20% → 25%
→ 0% → 0%` vs `wall_aware`). MCTS amplifies whatever the value head believes, so
more sims amplified a wrong belief.

**Therefore the gating metric for every phase is the sim-scaling curve
(`scripts/diagnose.py`), not iteration count or win-rate at a fixed depth.**
Spending money to reach huge sim counts before the value head scales is lighting
money on fire.

> Note on curriculum axis: a *smaller-board* curriculum (5×5 → 7×7) is **not**
> cheap here — `BOARD_SIZE`/`WALLS_PER_PLAYER` are hardcoded and the policy head
> is size-dependent, so weights don't transfer across sizes. The curriculum axis
> we actually use is **simulation count**: cheap, low-sim games early; expensive,
> deep games only once search is a positive improvement operator.

## The curriculum (with game counts)

Game counts follow the loop's `games ≈ ROUNDS × GAMES`. Targets track the
runbook's 200k–500k "competitive" band.

| Phase | What runs | Sims/move | Games | Advance when |
|---|---|---|---|---|
| **0. Warm-start** | supervised imitation of `wall_aware` (no MCTS) | 0 | ~5k playout games (~200k positions) | pretrain converges |
| **1. GATE** | `scripts/diagnose.py` — measurement only | 64 / 256 / 800 | ~60 | calib ≳75% **and** win-rate *climbs* with sims |
| **2. Low-sim loop** | self-play; verify the loop *compounds* | 64 → 128 | **~30k** (200 rounds × 150) | `wall_aware` rises & holds; re-gate stays green |
| **3. Main run** | build competitive strength | 400 → 600 | **~200k** (500 × 400) | `wall_aware` > 50% and stable |
| **4. Ceiling-break** | extract deep wall tactics | 800 | **+200k–500k** | strength still climbs with sims |
| **Deploy** | tournament play only | **10k–100k** | — | — |

Total ≈ **450k–750k self-play games** to competitive. On one modern GPU the bulk
(Phase 3) is ~1–3 days.

Two hard rules that fall out of this project's history:

- **Never raise sims while the sim-scaling curve is flat/humped.** Deeper search
  on a bad value head is the regression we already hit. Fix value → *then* ramp.
- **Keep anchoring + opponent diversity through every phase** (`ANCHOR`/anchor-frac
  decay, `--opponent-prob`). Pure self-play forgets and only learns to beat its
  own racer style.

## Deploy on a GPU server + launch

### 0. Provision + setup

Rent one GPU (RTX 4090 / L4 / A100; ~$0.2–0.6/hr on Vast.ai / RunPod / Lambda).
Prefer **on-demand** over spot for the first run.

```bash
git clone <your-fork> Quorridor && cd Quorridor
bash scripts/gpu_setup.sh          # Go + CUDA torch, builds Go, runs tests
. .venv/bin/activate
```

Confirm it prints `cuda available: True` before continuing.

### 1. Warm-start (Phase 0)

```bash
python -m quoridor.training.warmstart --out-dir data/warm --games 5000 --policy wall_aware
python -m quoridor.training.pretrain  --data-dir data/warm --out checkpoints/warm.pt \
    --channels 128 --blocks 12 --epochs 15 --value-weight 2.0 --device cuda
```

### 2. GATE (Phase 1) — do not skip

```bash
python scripts/diagnose.py --checkpoint checkpoints/warm.pt \
    --channels 128 --blocks 12 --device cuda --opponent wall_aware --games 60 \
    --sims 64 256 800
```

- **Green** = calibration ≳75% **and** win-rate climbs with sims → proceed.
- **Red** = fix the value signal first (bigger net, higher `--value-weight`, more
  warm-start data/epochs) and re-gate. Nothing downstream works until it's green.

### 3. Main loop (Phases 2–4) — crash-safe

Run under `tmux` (survives SSH drops) via the supervisor (survives crashes /
preemptions):

```bash
tmux new -s train
DEVICE=cuda CH=128 BL=12 SIMS=600 GAMES=400 CONCURRENCY=$(nproc) \
  EVAL_GAMES=60 ROUNDS=500 SNAPSHOT_EVERY=10 \
  WARM=checkpoints/warm.pt ANCHOR=data/warm \
  bash scripts/train_supervisor.sh 2>&1 | tee train.log
```

Detach from tmux with `Ctrl-b d`; reattach with `tmux attach -t train`.

Watch `[eval round N]`: **`wall_aware` should rise above 50% and hold.** Re-run
`scripts/diagnose.py` on `checkpoints/current.pt` every ~50 rounds for a
noise-free read.

### Recommended configs (estimates — iterate)

| Knob | Get competitive | Break ceiling / strong |
|---|---|---|
| Net size | 128ch × 12blk (~10M) | 192–256ch × 15–20blk (~15–40M) |
| Sims/move (train) | 400–600 | 600–800 |
| Self-play games | 200k–500k | 1–5M |
| ~1× modern GPU time | ~1–3 days | ~1–4 weeks |

## Crash safety — how it works

Two layers, so a cheap/spot GPU that gets reclaimed costs at most **one round**:

**1. Snapshots + progress (`go_train_loop.sh`).**
- After every round the last-completed round number is written to
  `checkpoints/snapshots/loop_state`.
- Every `SNAPSHOT_EVERY` rounds (default 10) `current.pt` is archived to
  `checkpoints/snapshots/round_N.pt` — your **rollback points** if a round
  regresses `wall_aware`.
- Any failed step — self-play, learner, or a dead inference server — aborts the
  script with a distinct non-zero exit code instead of silently continuing.

**2. Resume + auto-relaunch (`train_supervisor.sh`).**
- On relaunch the loop detects `loop_state` + `current.pt` and **resumes at the
  next round**, keeping the trained weights (it does *not* restart from `WARM`).
  The anchor-frac/LR decay schedule is a function of the absolute round number,
  so it stays consistent across a resume.
- The supervisor relaunches the loop after any non-zero exit with exponential
  backoff (`BACKOFF` 15s → `BACKOFF_MAX` 300s), up to `MAX_RETRIES` (default
  1000). It stops only on a clean `GO_LOOP_DONE` (exit 0).

### Force a clean restart

To start over from `WARM` instead of resuming:

```bash
FRESH=1 ... bash scripts/train_supervisor.sh    # FRESH applies to attempt 1 only
```

### Roll back a regression

If `wall_aware` drops and stays down, promote an earlier snapshot and continue:

```bash
# inspect snapshots; pick the last good one, e.g. round_120.pt
cp checkpoints/snapshots/round_120.pt checkpoints/current.pt
echo 120 > checkpoints/snapshots/loop_state
# relaunch: resumes at round 121 from the good weights
... bash scripts/train_supervisor.sh
```

### Knobs quick-reference (crash safety)

| Env | Default | Meaning |
|---|---|---|
| `SNAPSHOT_EVERY` | 10 | archive `current.pt` every N rounds |
| `SNAP_DIR` | `checkpoints/snapshots` | snapshots + resume state |
| `FRESH` | 0 | `1` = ignore prior progress, restart from `WARM` |
| `MAX_RETRIES` | 1000 | supervisor: give up after N failed relaunches |
| `BACKOFF` / `BACKOFF_MAX` | 15 / 300 | supervisor: retry backoff seconds |
