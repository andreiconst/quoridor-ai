"""Strength evaluation: arena matches against a fixed baseline.

The baseline is a deterministic shortest-path bot -- it always advances its
pawn along a current shortest route to its goal and never places walls. It is
weak, but it's a meaningful first yardstick: an agent that can't beat it hasn't
learned to use walls at all. Track the win rate over training to see real
progress (and when it plateaus).
"""

from __future__ import annotations

from collections import deque

from ..engine.game import apply_action, legal_action_mask, pawn_action
from ..engine.state import BOARD_SIZE, State
from .mcts import MCTS


def _distance_to_goal(state: State, player: int) -> dict:
    """BFS distance from every reachable cell to the player's goal row,
    respecting current walls."""
    goal_row = state.goal_row(player)
    dist = {}
    queue = deque()
    for c in range(BOARD_SIZE):
        cell = (goal_row, c)
        dist[cell] = 0
        queue.append(cell)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                state._in_bounds(nr, nc)
                and (nr, nc) not in dist
                and not state._edge_blocked(r, c, nr, nc)
            ):
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return dist


def shortest_path_action(state: State) -> int:
    """Baseline policy: step onto the legal pawn destination closest to goal."""
    player = state.current_player
    dist = _distance_to_goal(state, player)
    best_dest, best_d = None, float("inf")
    for dest in state.legal_pawn_destinations(player):
        d = dist.get(dest, float("inf"))
        if d < best_d:
            best_d, best_dest = d, dest
    if best_dest is None:  # no pawn move (shouldn't happen); pass via any legal action
        best_dest = state.pawns[player]
    return pawn_action(best_dest)


def mcts_action(mcts: MCTS, state: State) -> int:
    """Greedy (temperature 0) action from a network-guided MCTS search."""
    root = mcts.run(state, add_noise=False)
    policy = MCTS.policy_from_visits(root, temperature=0.0)
    return int(policy.argmax())


def play_match(action_fns, max_moves: int = 200) -> int:
    """Play one game; action_fns[player] -> action int. Returns winner or -1."""
    state = State.initial()
    for _ in range(max_moves):
        if state.is_terminal() or legal_action_mask(state).sum() == 0:
            break
        action = action_fns[state.current_player](state)
        state = apply_action(state, action)
    return state.winner


def evaluate_vs_baseline(
    network,
    n_games: int = 20,
    num_simulations: int = 100,
    device: str = "cpu",
    mcts_batch_size: int = 16,
):
    """Play the network (via MCTS) against the shortest-path baseline,
    alternating who moves first. Returns (wins, draws, losses) for the network.
    """
    mcts = MCTS(network, device=device, num_simulations=num_simulations,
                batch_size=mcts_batch_size)
    net_fn = lambda s: mcts_action(mcts, s)
    base_fn = lambda s: shortest_path_action(s)

    wins = draws = losses = 0
    for g in range(n_games):
        if g % 2 == 0:
            fns, net_player = [net_fn, base_fn], 0
        else:
            fns, net_player = [base_fn, net_fn], 1
        winner = play_match(fns)
        if winner == net_player:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses
