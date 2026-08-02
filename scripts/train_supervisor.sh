#!/bin/bash
# Crash-safe supervisor for the Go training loop.
#
# Relaunches scripts/go_train_loop.sh whenever it exits non-zero -- a step
# failure, an OOM, a dropped inference server, or a spot-instance preemption --
# with exponential backoff. Because go_train_loop.sh RESUMES from
# checkpoints/current.pt (see its FRESH/STATE_FILE logic), each relaunch picks
# up at the next unfinished round, so a crash costs at most one round of work.
#
# The loop prints "GO_LOOP_DONE" and exits 0 when all ROUNDS are complete; that
# is the only condition under which this supervisor stops.
#
# Usage:
#   DEVICE=cuda CH=128 BL=12 SIMS=600 GAMES=400 ROUNDS=500 \
#     WARM=checkpoints/warm.pt ANCHOR=data/warm \
#     bash scripts/train_supervisor.sh
#
# Recommended: run under tmux/nohup so it survives your SSH session dropping:
#   tmux new -s train
#   ... bash scripts/train_supervisor.sh 2>&1 | tee train.log
set -u

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO"

MAX_RETRIES=${MAX_RETRIES:-1000}   # give up after this many failed relaunches
BACKOFF=${BACKOFF:-15}             # initial seconds to wait after a crash
BACKOFF_MAX=${BACKOFF_MAX:-300}    # cap the backoff here
LOG=${LOG:-train_supervisor.log}

# FRESH (if the user set it) applies to the FIRST attempt only; every relaunch
# after that must resume, never wipe. Capture and then neutralise it.
FRESH_FIRST=${FRESH:-0}

attempt=0
delay=$BACKOFF
while :; do
  attempt=$(( attempt + 1 ))

  if [ "$attempt" -eq 1 ]; then export FRESH="$FRESH_FIRST"; else export FRESH=0; fi

  echo "=== [supervisor] attempt $attempt (FRESH=$FRESH) at $(date -u +%FT%TZ) ===" | tee -a "$LOG"

  bash scripts/go_train_loop.sh 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}

  if [ "$code" -eq 0 ]; then
    echo "=== [supervisor] training completed cleanly after $attempt attempt(s) ===" | tee -a "$LOG"
    break
  fi

  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "=== [supervisor] GIVING UP after $attempt attempts (last exit=$code) ===" | tee -a "$LOG"
    exit "$code"
  fi

  echo "=== [supervisor] loop exited $code; resuming in ${delay}s (attempt $((attempt+1))) ===" | tee -a "$LOG"
  sleep "$delay"
  delay=$(( delay * 2 )); [ "$delay" -gt "$BACKOFF_MAX" ] && delay=$BACKOFF_MAX
done
