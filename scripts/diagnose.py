"""The gate before scaling: is the AlphaZero improvement operator working?

Two checks on a checkpoint:
  1. Value-head calibration -- does the value predict who actually wins?
  2. Sim-scaling -- does win-rate vs a baseline CLIMB with more MCTS sims?

If calibration is weak (<~75%) OR win-rate does NOT climb with sims, the loop
will NOT compound no matter how many games you throw at it -- fix the value
signal first. Only scale games once both are green. (This is the single lesson
that would have saved the most time in the laptop phase.)

  python scripts/diagnose.py --checkpoint checkpoints/current.pt \
      --channels 128 --blocks 10 --device cuda --opponent wall_aware --games 40
"""

import argparse
import time

import numpy as np
import torch

from quoridor.training.network import QuoridorNet
from quoridor.training.evaluate import BASELINES, mcts_action, random_opening
from quoridor.training.mcts import MCTS
from quoridor.engine.game import apply_action, legal_action_mask, encode_state, encode_action_for_player
from quoridor.engine.state import State


def calibration(net, device, opp_fn, n_games, open_plies):
    rng = np.random.default_rng(1)
    v, o = [], []
    for _ in range(n_games):
        s = State.initial()
        for _ in range(int(rng.integers(0, open_plies + 1))):
            s = apply_action(s, int(rng.choice(np.nonzero(legal_action_mask(s))[0])))
            if s.is_terminal():
                s = State.initial(); break
        states = []
        for _ in range(200):
            if s.is_terminal() or legal_action_mask(s).sum() == 0:
                break
            states.append((s.clone(), s.current_player))
            s = apply_action(s, opp_fn(s) if rng.random() > 0.12 else int(rng.choice(np.nonzero(legal_action_mask(s))[0])))
        winner = s.winner
        for st, pl in states:
            with torch.no_grad():
                _, val = net.predict(torch.from_numpy(encode_state(st)).to(device))
            v.append(float(val)); o.append(0.0 if winner == -1 else (1.0 if pl == winner else -1.0))
    v, o = np.array(v), np.array(o); m = o != 0
    return (np.sign(v[m]) == np.sign(o[m])).mean(), np.corrcoef(v[m], o[m])[0, 1]


def raw_action(net, device, s):
    with torch.no_grad():
        probs, _ = net.predict(torch.from_numpy(encode_state(s)).to(device))
    probs = probs.cpu().numpy(); mask = legal_action_mask(s); best, bp = -1, -1.0
    for a in np.nonzero(mask)[0]:
        p = probs[encode_action_for_player(s, int(a))]
        if p > bp:
            bp, best = p, int(a)
    return best


def winrate(net, device, opp_fn, action_fn, n_games, open_plies):
    rng = np.random.default_rng(0); w = d = l = 0
    for g in range(n_games):
        s = State.initial()
        for _ in range(int(rng.integers(0, open_plies + 1))):
            s = apply_action(s, int(rng.choice(np.nonzero(legal_action_mask(s))[0])))
            if s.is_terminal():
                s = State.initial(); break
        npl = g % 2
        for _ in range(200):
            if s.is_terminal() or legal_action_mask(s).sum() == 0:
                break
            s = apply_action(s, action_fn(s) if s.current_player == npl else opp_fn(s))
        if s.winner == npl:
            w += 1
        elif s.winner == -1:
            d += 1
        else:
            l += 1
    return w, d, l


def main():
    p = argparse.ArgumentParser(description="Improvement-operator diagnostic.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--opponent", default="wall_aware", choices=list(BASELINES))
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sims", type=int, nargs="+", default=[64, 256, 800])
    p.add_argument("--open-plies", type=int, default=6)
    args = p.parse_args()

    net = QuoridorNet(channels=args.channels, num_blocks=args.blocks)
    net.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    net.to(args.device).eval()
    opp = BASELINES[args.opponent]

    print(f"=== diagnostic: {args.checkpoint} vs {args.opponent} ({args.games} games) ===", flush=True)
    acc, corr = calibration(net, args.device, opp, args.games, args.open_plies)
    verdict = "OK" if acc >= 0.75 else "WEAK -> fix value signal first"
    print(f"1) value calibration: sign-accuracy {acc:.0%}, corr {corr:.2f}  [{verdict}]", flush=True)

    print("2) sim-scaling (win-rate should CLIMB with sims):", flush=True)
    t = time.time()
    w, d, l = winrate(net, args.device, opp, lambda s: raw_action(net, args.device, s), args.games, args.open_plies)
    print(f"     raw (0 sims): {w/args.games:.0%}  ({w}W-{d}D-{l}L)  [{time.time()-t:.0f}s]", flush=True)
    for sims in args.sims:
        t = time.time()
        fn = lambda s, _s=sims: mcts_action(MCTS(net, device=args.device, num_simulations=_s, batch_size=16), s)
        w, d, l = winrate(net, args.device, opp, fn, args.games, args.open_plies)
        print(f"     {sims:4d} sims: {w/args.games:.0%}  ({w}W-{d}D-{l}L)  [{time.time()-t:.0f}s]", flush=True)
    print("GATE: scale games only if calibration >=~75% AND win-rate rises with sims.", flush=True)


if __name__ == "__main__":
    main()
