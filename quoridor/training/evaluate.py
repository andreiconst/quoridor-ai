"""Strength evaluation: arena matches against a fixed baseline.

The baseline is a deterministic shortest-path bot -- it always advances its
pawn along a current shortest route to its goal and never places walls. It is
weak, but it's a meaningful first yardstick: an agent that can't beat it hasn't
learned to use walls at all. Track the win rate over training to see real
progress (and when it plateaus).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..engine.game import apply_action, legal_action_mask, pawn_action, wall_action
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


def random_action(state: State) -> int:
    """Baseline policy: uniformly random legal action (weak; improvement vs this
    shows up early, before the net can beat the goal-rushing shortest-path bot)."""
    legal = np.nonzero(legal_action_mask(state))[0]
    return int(np.random.choice(legal))


def wall_aware_action(state: State) -> int:
    """Stronger baseline: race toward goal, but when not ahead, place the wall
    that sets the opponent back the most (net of self-cost). A meaningful
    absolute yardstick once the net can already race."""
    player = state.current_player
    opp = 1 - player
    own_d = _dist(state, player)
    opp_d = _dist(state, opp)

    if state.walls_left[player] > 0 and own_d >= opp_d:
        best_gain, best_wall = 0, None
        for r, c, orientation in state.legal_wall_slots(player):
            state.wall_slots[r][c] = orientation
            gain = (_dist(state, opp) - opp_d) - (_dist(state, player) - own_d)
            state.wall_slots[r][c] = 0
            if gain > best_gain:
                best_gain, best_wall = gain, (r, c, orientation)
        if best_wall is not None and best_gain >= 1:
            return wall_action(*best_wall)
    return shortest_path_action(state)


def _dist(state: State, player: int) -> int:
    d = _distance_to_goal(state, player)
    return d.get(state.pawns[player], 10**6)


BASELINES = {
    "random": random_action,
    "shortest_path": shortest_path_action,
    "wall_aware": wall_aware_action,
}


def mcts_action(mcts: MCTS, state: State) -> int:
    """Greedy (temperature 0) action from a network-guided MCTS search."""
    root = mcts.run(state, add_noise=False)
    policy = MCTS.policy_from_visits(root, temperature=0.0)
    return int(policy.argmax())


def random_opening(open_plies: int) -> State:
    """A start position after up to `open_plies` random legal moves. Diversifies
    games so deterministic players produce independent outcomes (without this,
    two deterministic players replay the same game every time)."""
    state = State.initial()
    k = int(np.random.randint(0, open_plies + 1)) if open_plies > 0 else 0
    for _ in range(k):
        legal = np.nonzero(legal_action_mask(state))[0]
        if len(legal) == 0:
            break
        state = apply_action(state, int(np.random.choice(legal)))
        if state.is_terminal():
            return State.initial()
    return state


def play_match(action_fns, max_moves: int = 200, start: State | None = None) -> int:
    """Play one game; action_fns[player] -> action int. Returns winner or -1."""
    state = start.clone() if start is not None else State.initial()
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
    opponent: str = "shortest_path",
    open_plies: int = 6,
):
    """Play the network (via MCTS) against a baseline, alternating who moves
    first, each game from a random opening so results are independent (both
    players are deterministic). Returns (wins, draws, losses) for the network.
    """
    mcts = MCTS(network, device=device, num_simulations=num_simulations,
                batch_size=mcts_batch_size)
    net_fn = lambda s: mcts_action(mcts, s)
    base_fn = BASELINES[opponent]

    wins = draws = losses = 0
    for g in range(n_games):
        if g % 2 == 0:
            fns, net_player = [net_fn, base_fn], 0
        else:
            fns, net_player = [base_fn, net_fn], 1
        winner = play_match(fns, start=random_opening(open_plies))
        if winner == net_player:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses


def evaluate_vs_net(
    network,
    opponent,
    n_games: int = 20,
    num_simulations: int = 100,
    device: str = "cpu",
    mcts_batch_size: int = 16,
    open_plies: int = 6,
):
    """Head-to-head arena: `network` vs a frozen `opponent` net, both via MCTS,
    alternating first move, from random openings. Returns (wins, draws, losses)
    for `network`. The non-saturating "am I better than my past self" metric."""
    mcts_a = MCTS(network, device=device, num_simulations=num_simulations, batch_size=mcts_batch_size)
    mcts_b = MCTS(opponent, device=device, num_simulations=num_simulations, batch_size=mcts_batch_size)
    a_fn = lambda s: mcts_action(mcts_a, s)
    b_fn = lambda s: mcts_action(mcts_b, s)

    wins = draws = losses = 0
    for g in range(n_games):
        if g % 2 == 0:
            fns, a_player = [a_fn, b_fn], 0
        else:
            fns, a_player = [b_fn, a_fn], 1
        winner = play_match(fns, start=random_opening(open_plies))
        if winner == a_player:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses
