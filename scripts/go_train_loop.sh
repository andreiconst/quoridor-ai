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

rm -rf "$DATA"; mkdir -p "$DATA"
cp "$WARM" checkpoints/current.pt

echo "=== build go self-play binary ==="
(cd go && $GO build -o /tmp/go_loop_sp ./cmd/selfplay) || exit 1

echo "=== start inference server (device=$DEVICE, ${CH}ch x ${BL}blk) ==="
$PY -m quoridor.serving.infer_server --socket "$SOCK" --checkpoint checkpoints/current.pt \
    --device "$DEVICE" --channels $CH --blocks $BL > /tmp/go_loop_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f $SOCK' EXIT
for i in $(seq 1 100); do [ -S "$SOCK" ] && break; sleep 0.2; done

for r in $(seq 1 "$ROUNDS"); do
  t0=$(date +%s)
  FRAC=$(awk "BEGIN{p=($r-1)/($ROUNDS-1); f=$ANCHOR_HI-($ANCHOR_HI-$ANCHOR_LO)*p; printf \"%.3f\", f}")
  LR=$(awk "BEGIN{p=($r-1)/($ROUNDS-1); l=$LR_HI*exp(log($LR_LO/$LR_HI)*p); printf \"%.7f\", l}")
  /tmp/go_loop_sp --socket "$SOCK" --games $GAMES --concurrency $CONCURRENCY --sims $SIMS \
      --batch 16 --linger-us 1000 --data-dir "$DATA" >/dev/null 2>&1
  $PY -m quoridor.serving.learner --data-dir "$DATA" --out checkpoints/current.pt \
      --channels $CH --blocks $BL --resume checkpoints/current.pt \
      --anchor-data "$ANCHOR" --anchor-frac $FRAC --steps $LEARN_STEPS --lr $LR \
      --device "$DEVICE" 2>&1 | grep -E "trained|anchoring" | tail -1
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
echo "GO_LOOP_DONE"
