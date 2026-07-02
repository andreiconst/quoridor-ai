#!/bin/bash
# Go+MPS scale-up training loop for the 3M net.
#
# Architecture (see docs/PROTOCOL.md): a Python MPS inference server serves
# current.pt; Go runs parallel self-play against it writing shards; the Python
# learner trains on the shards (anchored on the warm-start data) and atomically
# republishes current.pt, which the server hot-reloads. Repeat.
#
# Usage: bash scripts/go_train_loop.sh [rounds]
set -u
cd /Users/andreiconstantinescu/Repos/Quorridor
PY=.venv/bin/python
GO=/opt/homebrew/bin/go

CH=128; BL=10                  # 3M net
ROUNDS=${1:-60}
GAMES=200                      # self-play games per round
SIMS=128
LEARN_STEPS=300
EVAL_EVERY=3
SOCK=/tmp/go_loop.sock
DATA=/tmp/go_loop_data
ANCHOR=/tmp/qwall             # wall-aware warm-start data (anchor)
WARM=checkpoints/warm3m.pt

rm -rf "$DATA"; mkdir -p "$DATA"
cp "$WARM" checkpoints/current.pt

echo "=== build go self-play binary ==="
(cd go && $GO build -o /tmp/go_loop_sp ./cmd/selfplay) || exit 1

echo "=== start MPS inference server (3M) ==="
$PY -m quoridor.serving.infer_server --socket "$SOCK" --checkpoint checkpoints/current.pt \
    --device mps --channels $CH --blocks $BL > /tmp/go_loop_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f $SOCK' EXIT
for i in $(seq 1 100); do [ -S "$SOCK" ] && break; sleep 0.2; done

for r in $(seq 1 "$ROUNDS"); do
  t0=$(date +%s)
  /tmp/go_loop_sp --socket "$SOCK" --games $GAMES --concurrency 12 --sims $SIMS \
      --batch 16 --linger-us 1000 --data-dir "$DATA" >/dev/null 2>&1
  $PY -m quoridor.serving.learner --data-dir "$DATA" --out checkpoints/current.pt \
      --channels $CH --blocks $BL --resume checkpoints/current.pt \
      --anchor-data "$ANCHOR" --anchor-frac 0.25 --steps $LEARN_STEPS --lr 3e-4 \
      --device mps 2>&1 | grep -E "trained|anchoring" | tail -1
  echo "[round $r/$ROUNDS] done in $(( $(date +%s) - t0 ))s"

  if [ $(( r % EVAL_EVERY )) -eq 0 ]; then
    $PY -c "
import torch
from quoridor.training.network import QuoridorNet
from quoridor.training.evaluate import evaluate_vs_baseline
net=QuoridorNet(channels=$CH,num_blocks=$BL); net.load_state_dict(torch.load('checkpoints/current.pt',map_location='cpu')); net.eval()
out=[]
for opp in ['shortest_path','wall_aware']:
    w,d,l=evaluate_vs_baseline(net,n_games=12,num_simulations=$SIMS,opponent=opp)
    out.append(f'{opp}={w/12:.0%} ({w}W-{d}D-{l}L)')
print('    [eval round $r] '+' | '.join(out))
"
  fi
done
echo "GO_LOOP_DONE"
