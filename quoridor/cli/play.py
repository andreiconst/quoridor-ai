"""Play against a trained (or untrained, randomly-initialized) agent in the terminal.

Move notation:
  - Pawn move: a letter+digit cell coordinate to move TO, e.g. "e5".
  - Wall:      "<col><row><h|v>" giving the wall slot's top-left cell and
               orientation, e.g. "e3h" or "c4v". Columns a-h, rows 1-8.
"""

from __future__ import annotations

import argparse

import torch

from ..engine.game import legal_action_mask, pawn_action, wall_action
from ..engine.state import BOARD_SIZE, WALL_GRID, WALL_H, WALL_V, State
from ..training.mcts import MCTS
from ..training.network import QuoridorNet

COLS = "abcdefghi"[:BOARD_SIZE]


def render(state: State) -> str:
    lines = []
    p0, p1 = state.pawns
    header = "   " + " ".join(COLS)
    lines.append(header)
    for r in range(BOARD_SIZE):
        row_cells = []
        for c in range(BOARD_SIZE):
            if (r, c) == p0:
                row_cells.append("0")
            elif (r, c) == p1:
                row_cells.append("1")
            else:
                row_cells.append(".")
        row_str = f"{r:2d} " + " ".join(row_cells)
        # vertical wall indicators to the right of each cell
        wall_row = []
        for c in range(BOARD_SIZE):
            if c < WALL_GRID and r < WALL_GRID and state.wall_slots[r][c] == WALL_V:
                wall_row.append("|")
            elif c > 0 and c - 1 < WALL_GRID and r < WALL_GRID and state.wall_slots[r][c - 1] == WALL_V:
                wall_row.append("|")
            else:
                wall_row.append(" ")
        lines.append(row_str)
        if r < WALL_GRID:
            below = ["   "]
            for c in range(BOARD_SIZE):
                mark = " "
                for cc in (c - 1, c):
                    if 0 <= cc < WALL_GRID and state.wall_slots[r][cc] == WALL_H:
                        mark = "-"
                below.append(mark)
            lines.append(" ".join(below))
    lines.append(f"Player 0 walls left: {state.walls_left[0]}  Player 1 walls left: {state.walls_left[1]}")
    return "\n".join(lines)


def parse_cell(token: str):
    col = COLS.index(token[0])
    row = int(token[1:])
    return row, col


def parse_move(token: str, state: State):
    token = token.strip().lower()
    if token[-1] in ("h", "v"):
        col = COLS[: WALL_GRID + 1].index(token[0])
        row = int(token[1:-1])
        orientation = WALL_H if token[-1] == "h" else WALL_V
        return wall_action(row, col, orientation)
    row, col = parse_cell(token)
    return pawn_action((row, col))


def load_network(checkpoint: str | None, device: str):
    network = QuoridorNet().to(device)
    if checkpoint:
        network.load_state_dict(torch.load(checkpoint, map_location=device))
        network.eval()
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        network.eval()
        print("No checkpoint given; playing against an untrained (random) network.")
    return network


def main():
    parser = argparse.ArgumentParser(description="Play Quoridor against the AI.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained model .pt file")
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations per AI move")
    parser.add_argument("--human-player", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    network = load_network(args.checkpoint, args.device)
    mcts = MCTS(network, device=args.device, num_simulations=args.simulations)

    state = State.initial()
    print(render(state))

    while not state.is_terminal():
        if legal_action_mask(state).sum() == 0:
            print("No legal moves available; stopping.")
            break

        if state.current_player == args.human_player:
            move = input(f"\nPlayer {state.current_player} move (e.g. e5 or e3h): ").strip()
            try:
                action = parse_move(move, state)
            except (ValueError, IndexError):
                print("Could not parse that move, try again.")
                continue
            mask = legal_action_mask(state)
            if action >= len(mask) or mask[action] == 0:
                print("Illegal move, try again.")
                continue
            from ..engine.game import apply_action

            state = apply_action(state, action)
        else:
            print("\nAI is thinking...")
            root = mcts.run(state, add_noise=False)
            policy = MCTS.policy_from_visits(root, temperature=0.0)
            action = int(policy.argmax())
            state = root.children[action].state
            print(f"AI plays action {action}")

        print(render(state))

    if state.winner != -1:
        print(f"\nPlayer {state.winner} wins!")
    else:
        print("\nGame ended with no winner (move limit or no legal moves).")


if __name__ == "__main__":
    main()
