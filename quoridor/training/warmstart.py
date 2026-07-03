"""Generate warm-start data: racing games to teach the net to reach its goal.

Self-play from a random net can't bootstrap goal-reaching (the cold-start trap).
Here we generate supervised data where the *target* policy at every state is the
shortest-path advancing move, while the *played* move adds exploration (random
pawn moves / occasional walls) so the states are diverse. Pretraining on this
makes the net race competently from the start; self-play then learns wall play.

    python -m quoridor.training.warmstart --out-dir data/warm --games 3000
    python -m quoridor.training.pretrain --data-dir data/warm --out checkpoints/warm.pt
    python -m quoridor.training.train --resume checkpoints/warm.pt ...
"""

from __future__ import annotations

import argparse

import numpy as np

from ..engine.game import (
    ACTION_SIZE,
    apply_action,
    encode_action_for_player,
    encode_state,
    legal_action_mask,
    pawn_action,
    wall_action,
)
from ..engine.state import State
from .dataset import ShardWriter
from .evaluate import BASELINES
from .selfplay import MAX_MOVES, _winner_by_progress


def play_racing_game(rng, target_fn, open_plies: int = 8):
    """Random opening (for diversity), then DETERMINISTIC target_fn play to the
    end. Each recorded (tail) state gets policy = target_fn's one-hot move and
    value = the deterministic playout's outcome from that state -- i.e. "who
    wins from here under strong play". This gives a clean position-value signal,
    unlike labeling every state of a noisy game with the final (tempo-decided)
    outcome, which left the value head near-random (64% sign accuracy)."""
    state = State.initial()

    # Unrecorded random opening for state diversity.
    for _ in range(int(rng.integers(0, open_plies + 1))):
        legal = np.nonzero(legal_action_mask(state))[0]
        if len(legal) == 0:
            break
        state = apply_action(state, int(rng.choice(legal)))
        if state.is_terminal():
            return []  # opening decided it; nothing clean to learn

    # Deterministic playout; the tail from any state IS the rollout from it.
    examples = []
    for _ in range(MAX_MOVES):
        if legal_action_mask(state).sum() == 0:
            break
        player = state.current_player
        target = target_fn(state)
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        policy[encode_action_for_player(state, target)] = 1.0
        examples.append((encode_state(state), policy, player))
        state = apply_action(state, target)
        if state.is_terminal():
            break

    winner = state.winner
    if winner == -1:
        winner = _winner_by_progress(state)
    out = []
    for planes, policy, player in examples:
        outcome = 0.0 if winner == -1 else (1.0 if player == winner else -1.0)
        out.append((planes, policy, outcome))
    return out


def generate(out_dir: str, n_games: int, policy: str = "shortest_path", seed: int = 0, shard_size: int = 50000):
    target_fn = BASELINES[policy]
    rng = np.random.default_rng(seed)
    writer = ShardWriter(out_dir, shard_size=shard_size)
    for g in range(n_games):
        for planes, pol, outcome in play_racing_game(rng, target_fn):
            writer.add(planes, pol, outcome)
        if (g + 1) % 200 == 0:
            print(f"  {g + 1}/{n_games} games, {writer.total_written + len(writer._values)} examples", flush=True)
    writer.close()
    print(f"wrote {writer.total_written} examples to {out_dir}", flush=True)
    return writer.total_written


def main():
    p = argparse.ArgumentParser(description="Generate warm-start data by imitating a baseline.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--policy", choices=list(BASELINES), default="shortest_path",
                   help="Which baseline to imitate (shortest_path races; wall_aware also blocks)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.out_dir, args.games, policy=args.policy, seed=args.seed)


if __name__ == "__main__":
    main()
