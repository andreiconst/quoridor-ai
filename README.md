# Quorridor

An AlphaZero-style AI for Quoridor (standard 9x9 board, 2 players, 10 walls each),
written in Go (also python version) with Inference in Pytorch.

## Layout

```
quoridor/
  engine/
    state.py     # core rules: pawn moves/jumps, wall legality, win condition
    game.py      # fixed-size action space (209 actions) + NN board encoding
  training/
    network.py   # residual CNN with policy + valuv
    mcts.py       # PUCT MCTS guided by the network
    selfplay.py   # self-play game generation -> training examples
    train.py      # self-play / train loop with checkpointing
  cli/
    play.py       # terminal interface to play against a trained agent
tests/
  test_engine.py  # rules engine unit tests
checkpoints/        # saved model weights (git-ignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest tests/ -q
```

## Train

```bash
python -m quoridor.training.train \
  --iterations 50 \
  --games-per-iteration 20 \
  --num-simulations 20 --batch-size 32 --workers 4
```

Each iteration: plays self-play games with MCTS guided by the current
network, trains on a replay buffer of recent positions, and saves a
checkpoint to `checkpoints/latest.pt` (plus `checkpoints/model_iter<N>.pt`).

This is compute-hungry by AlphaZero tradition — meaningful play strength
needs many iterations and simulations. Start small (few iterations, low
simulation count) to confirm the pipeline runs, then scale up. A GPU
(`--device cuda`) helps a lot once you increase model size/simulations.

## Play against the agent

```bash
python -m quoridor.cli.play --checkpoint checkpoints/latest.pt --simulations 400
```

Without `--checkpoint` it plays against an untrained (randomly initialized)
network, useful for sanity-checking the CLI itself.

Move notation:
- Pawn move: destination cell, e.g. `e5` (columns a-i, rows 0-8).
- Wall: `<col><row><h|v>` for the wall slot's top-left cell and orientation,
  e.g. `e3h` (horizontal) or `c4v` (vertical). Columns a-h, rows 0-7.

## How it works

- **Engine** (`engine/state.py`): full Quoridor rules including straight and
  diagonal pawn jumps, and wall-placement legality checked via BFS so no wall
  can fully seal off either player's goal.
- **Action space** (`engine/game.py`): a fixed 209-dim action vector — 81
  pawn-destination actions + 64 horizontal-wall + 64 vertical-wall actions —
  so the network has a constant-size policy head. Board state is encoded as
  6 planes (own pawn, opponent pawn, horizontal walls, vertical walls, own/
  opponent walls remaining) and flipped vertically when it's player 1's turn,
  so the network only ever has to learn to play as "the player moving down".
- **MCTS** (`training/mcts.py`): standard PUCT search using the network's
  policy as priors and its value head instead of rollouts.
- **Self-play loop** (`training/selfplay.py`, `training/train.py`): generates
  games via MCTS self-play, stores `(state, visit-count policy, outcome)`
  triples in a replay buffer, and trains the network to predict both.
