#!/bin/bash
# Go + GPU/MPS scale-up training loop.
#
# Architecture (see docs/PROTOCOL.md): a Python inference server serves
# current.pt on the accelerator; Go runs parallel self-play against it writing
# shards; the Python learner trains on the shards (anchored on warm-start data)
# and atomically republishes current.pt, which the server hot-reloads. Repeat.
#
# Everything is env-overridable so the same script runs on a laptop (MPS) or a
# cloud GPU (CUDA). See docs/GPU_RUNBOOK.md for recommended configs.
#
# Crash safety:
#   * Every round writes the last-completed round number to $STATE_FILE and
#     (every $SNAPSHOT_EVERY rounds) archives current.pt to $SNAP_DIR/round_N.pt
#     so a regression can be rolled back.
#   * On relaunch the loop RESUMES from checkpoints/current.pt at the next round
#     instead of restarting from $WARM -- so a spot-instance preemption costs at
#     most one round. Set FRESH=1 to force a clean restart from $WARM.
#   * Any failed step (self-play, learner, dead inference server) aborts with a
#     non-zero exit so a supervisor (scripts/train_supervisor.sh) can relaunch.
#
#   Usage:  [ENV=...] bash scripts/go_train_loop.sh [rounds]
set -u

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"
PY=${PY:-.venv/bin/python}
GO=${GO:-$(command -v go)}

DEVICE=${DEVICE:-mps}            # cuda on a GPU box
CH=${CH:-128}; BL=${BL:-10}     # network width/depth (3M default; scale up on GPU)
ROUNDS=${1:-${ROUNDS:-60}}
GAMES=${GAMES:-150}             # self-play games per round
SIMS=${SIMS:-64}                # MCTS sims/move (raise to 400-800 on a GPU)
CONCURRENCY=${CONCURRENCY:-12}  # Go self-play actors (~= CPU cores)
LEARN_STEPS=${LEARN_STEPS:-300}
EVAL_EVERY=${EVAL_EVERY:-3}
EVAL_GAMES=${EVAL_GAMES:-12}    # use >=50 on a real run (12-game evals are noisy)
# Anchor-frac and LR both decay over rounds: early rounds anchor hard (protect
# the warm-start) with a moderate LR; late rounds anchor lightly so self-play
# can push past the teacher, at a low LR to stay stable.
ANCHOR_HI=${ANCHOR_HI:-0.30}; ANCHOR_LO=${ANCHOR_LO:-0.05}
LR_HI=${LR_HI:-3e-4};         LR_LO=${LR_LO:-7e-5}
SOCK=${SOCK:-/tmp/go_loop.sock}
DATA=${DATA:-/tmp/go_loop_data}
ANCHOR=${ANCHOR:-/tmp/qwall2}                    # warm-start data (anchor)
WARM=${WARM:-checkpoints/warm3m_v2.pt}           # starting checkpoint

# --- crash-safety knobs ------------------------------------------------------
SNAP_DIR=${SNAP_DIR:-checkpoints/snapshots}      # rollback archive + resume state
STATE_FILE=${STATE_FILE:-$SNAP_DIR/loop_state}   # holds last-completed round number
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-10}             # archive current.pt every N rounds
SHARD_WINDOW=${SHARD_WINDOW:-12}                 # keep this many recent shards (~rounds); bounds disk
OUTCOME_WEIGHT=${OUTCOME_WEIGHT:-0.6}            # value target = w*outcome + (1-w)*mcts_search_value
FRESH=${FRESH:-0}                                # FRESH=1 => ignore prior progress
mkdir -p "$SNAP_DIR"

# --- resume vs fresh start ---------------------------------------------------
START_ROUND=1
if [ "$FRESH" != "1" ] && [ -f "$STATE_FILE" ] && [ -f checkpoints/current.pt ]; then
  LAST_ROUND=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
  case "$LAST_ROUND" in ''|*[!0-9]*) LAST_ROUND=0 ;; esac
  START_ROUND=$(( LAST_ROUND + 1 ))
  echo "=== RESUME: last completed round=$LAST_ROUND, continuing at round $START_ROUND ==="
  echo "    (keeping checkpoints/current.pt; set FRESH=1 to restart from $WARM)"
  mkdir -p "$DATA"                               # keep any accumulated shards
else
  echo "=== FRESH start from $WARM ==="
  rm -rf "$DATA"; mkdir -p "$DATA"
  cp "$WARM" checkpoints/current.pt
  echo 0 > "$STATE_FILE"
fi

if [ "$START_ROUND" -gt "$ROUNDS" ]; then
  echo "Nothing to do: already completed $ROUNDS rounds. (set FRESH=1 to restart)"
  echo "GO_LOOP_DONE"; exit 0
fi

echo "=== build go self-play binary ==="
(cd go && $GO build -o /tmp/go_loop_sp ./cmd/selfplay) || exit 1

echo "=== start inference server (device=$DEVICE, ${CH}ch x ${BL}blk) ==="
$PY -m quoridor.serving.infer_server --socket "$SOCK" --checkpoint checkpoints/current.pt \
    --device "$DEVICE" --channels $CH --blocks $BL > /tmp/go_loop_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f $SOCK' EXIT
for i in $(seq 1 100); do [ -S "$SOCK" ] && break; sleep 0.2; done
if [ ! -S "$SOCK" ]; then
  echo "ERROR: inference server never created $SOCK; see /tmp/go_loop_server.log" >&2
  tail -20 /tmp/go_loop_server.log >&2; exit 2
fi

for r in $(seq "$START_ROUND" "$ROUNDS"); do
  t0=$(date +%s)

  # Bail if the inference server has died -- the supervisor will relaunch us.
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "[round $r] inference server (pid $SRV) died; see /tmp/go_loop_server.log" >&2
    tail -20 /tmp/go_loop_server.log >&2; exit 2
  fi

  FRAC=$(awk "BEGIN{p=($r-1)/($ROUNDS-1); f=$ANCHOR_HI-($ANCHOR_HI-$ANCHOR_LO)*p; printf \"%.3f\", f}")
  LR=$(awk "BEGIN{p=($r-1)/($ROUNDS-1); l=$LR_HI*exp(log($LR_LO/$LR_HI)*p); printf \"%.7f\", l}")

  # --- self-play (Go actors -> shards) ---
  if ! /tmp/go_loop_sp --socket "$SOCK" --games $GAMES --concurrency $CONCURRENCY --sims $SIMS \
      --batch 16 --linger-us 1000 --data-dir "$DATA" > /tmp/go_loop_sp.log 2>&1; then
    echo "[round $r] SELF-PLAY failed; see /tmp/go_loop_sp.log" >&2
    tail -20 /tmp/go_loop_sp.log >&2; exit 3
  fi

  # Bound disk: keep only the newest $SHARD_WINDOW shards (sliding replay window).
  # Shards are uniquely named per round, so mtime order == creation order.
  ls -t "$DATA"/go_*.qsh 2>/dev/null | tail -n +$((SHARD_WINDOW + 1)) | xargs -r rm -f

  # --- learn (anchored) -> republish current.pt (atomic, hot-reloaded) ---
  if ! $PY -m quoridor.serving.learner --data-dir "$DATA" --out checkpoints/current.pt \
      --channels $CH --blocks $BL --resume checkpoints/current.pt \
      --anchor-data "$ANCHOR" --anchor-frac $FRAC --steps $LEARN_STEPS --lr $LR \
      --outcome-weight $OUTCOME_WEIGHT --device "$DEVICE" > /tmp/go_loop_learn.log 2>&1; then
    echo "[round $r] LEARNER failed; see /tmp/go_loop_learn.log" >&2
    tail -20 /tmp/go_loop_learn.log >&2; exit 4
  fi
  grep -E "trained|anchoring" /tmp/go_loop_learn.log | tail -1

  # Round fully completed: record progress (enables resume) and archive.
  echo "$r" > "$STATE_FILE"
  if [ $(( r % SNAPSHOT_EVERY )) -eq 0 ]; then
    cp checkpoints/current.pt "$SNAP_DIR/round_${r}.pt"
    echo "  [snapshot] $SNAP_DIR/round_${r}.pt"
  fi

  echo "[round $r/$ROUNDS] done in $(( $(date +%s) - t0 ))s (anchor=$FRAC lr=$LR)"

  if [ $(( r % EVAL_EVERY )) -eq 0 ]; then
    $PY -c "
import torch
from quoridor.training.network import QuoridorNet
from quoridor.training.evaluate import evaluate_vs_baseline
dev='$DEVICE'
net=QuoridorNet(channels=$CH,num_blocks=$BL); net.load_state_dict(torch.load('checkpoints/current.pt',map_location=dev)); net.to(dev).eval()
out=[]
for opp in ['shortest_path','wall_aware']:
    w,d,l=evaluate_vs_baseline(net,n_games=$EVAL_GAMES,num_simulations=$SIMS,device=dev,opponent=opp)
    out.append(f'{opp}={w/$EVAL_GAMES:.0%} ({w}W-{d}D-{l}L)')
print('    [eval round $r] '+' | '.join(out))
"
  fi
done

# Always archive the final net for easy retrieval / rollback baseline.
cp checkpoints/current.pt "$SNAP_DIR/round_${ROUNDS}.pt" 2>/dev/null || true
echo "GO_LOOP_DONE"
