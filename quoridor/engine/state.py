"""Core Quoridor rules engine: 9x9 board, 2 players, 10 walls each.

Coordinates: (row, col), row 0 is player 0's start side, row 8 is player 1's
start side. Player 0 goal is row 8, player 1 goal is row 0.

Walls live on an 8x8 grid of "slots". Slot (r, c) sits at the intersection of
cell rows r/r+1 and cell columns c/c+1. A horizontal wall at slot (r, c)
blocks the edges (r,c)-(r+1,c) and (r,c+1)-(r+1,c+1). A vertical wall at slot
(r, c) blocks the edges (r,c)-(r,c+1) and (r+1,c)-(r+1,c+1). A slot holds at
most one wall (of either orientation), which also prevents crossing walls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

BOARD_SIZE = 9
WALL_GRID = BOARD_SIZE - 1
WALLS_PER_PLAYER = 10

EMPTY, WALL_H, WALL_V = 0, 1, 2

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class State:
    pawns: list  # [(r0, c0), (r1, c1)]
    walls_left: list  # [int, int]
    wall_slots: list = field(default_factory=lambda: [[EMPTY] * WALL_GRID for _ in range(WALL_GRID)])
    current_player: int = 0
    winner: int = -1

    @staticmethod
    def initial() -> "State":
        return State(
            pawns=[(0, BOARD_SIZE // 2), (BOARD_SIZE - 1, BOARD_SIZE // 2)],
            walls_left=[WALLS_PER_PLAYER, WALLS_PER_PLAYER],
        )

    def clone(self) -> "State":
        return State(
            pawns=list(self.pawns),
            walls_left=list(self.walls_left),
            wall_slots=[row[:] for row in self.wall_slots],
            current_player=self.current_player,
            winner=self.winner,
        )

    def opponent(self) -> int:
        return 1 - self.current_player

    def goal_row(self, player: int) -> int:
        return BOARD_SIZE - 1 if player == 0 else 0

    def is_terminal(self) -> bool:
        return self.winner != -1

    # ---- movement helpers ----

    def _edge_blocked(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """True if a wall blocks the direct edge between two adjacent cells."""
        if r1 == r2:  # horizontal move (left/right)
            c = min(c1, c2)
            for rr in (r1 - 1, r1):
                if 0 <= rr < WALL_GRID and 0 <= c < WALL_GRID and self.wall_slots[rr][c] == WALL_V:
                    return True
            return False
        else:  # vertical move (up/down)
            r = min(r1, r2)
            for cc in (c1 - 1, c1):
                if 0 <= r < WALL_GRID and 0 <= cc < WALL_GRID and self.wall_slots[r][cc] == WALL_H:
                    return True
            return False

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def legal_pawn_destinations(self, player: int | None = None) -> list:
        """All cells the given player's pawn may legally move to this turn."""
        if player is None:
            player = self.current_player
        r, c = self.pawns[player]
        opp_r, opp_c = self.pawns[1 - player]
        dests = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not self._in_bounds(nr, nc) or self._edge_blocked(r, c, nr, nc):
                continue
            if (nr, nc) == (opp_r, opp_c):
                # straight jump over opponent
                jr, jc = nr + dr, nc + dc
                if self._in_bounds(jr, jc) and not self._edge_blocked(nr, nc, jr, jc):
                    dests.append((jr, jc))
                else:
                    # diagonal jumps
                    for sdr, sdc in DIRECTIONS:
                        if (sdr, sdc) in ((dr, dc), (-dr, -dc)):
                            continue
                        sr, sc = nr + sdr, nc + sdc
                        if (
                            self._in_bounds(sr, sc)
                            and not self._edge_blocked(nr, nc, sr, sc)
                        ):
                            dests.append((sr, sc))
            else:
                dests.append((nr, nc))
        return dests

    def move_pawn(self, player: int, dest) -> None:
        self.pawns[player] = dest
        if dest[0] == self.goal_row(player):
            self.winner = player
        self.current_player = 1 - self.current_player

    # ---- wall helpers ----

    def _path_exists(self, player: int) -> bool:
        """BFS from the pawn to its goal row. Inlined edge checks (no per-step
        helper calls) since this runs on the hot path of wall legality."""
        ws = self.wall_slots
        goal_row = self.goal_row(player)
        sr, sc = self.pawns[player]
        start = sr * BOARD_SIZE + sc
        visited = bytearray(BOARD_SIZE * BOARD_SIZE)
        visited[start] = 1
        queue = [start]
        qi = 0
        while qi < len(queue):
            cell = queue[qi]
            qi += 1
            r, c = divmod(cell, BOARD_SIZE)
            if r == goal_row:
                return True
            # up (r-1, c): blocked by a horizontal wall on slot-row r-1
            if r > 0:
                rr = r - 1
                if not ((c > 0 and ws[rr][c - 1] == WALL_H) or (c < WALL_GRID and ws[rr][c] == WALL_H)):
                    n = cell - BOARD_SIZE
                    if not visited[n]:
                        visited[n] = 1
                        queue.append(n)
            # down (r+1, c): blocked by a horizontal wall on slot-row r
            if r < BOARD_SIZE - 1:
                if not ((c > 0 and ws[r][c - 1] == WALL_H) or (c < WALL_GRID and ws[r][c] == WALL_H)):
                    n = cell + BOARD_SIZE
                    if not visited[n]:
                        visited[n] = 1
                        queue.append(n)
            # left (r, c-1): blocked by a vertical wall on slot-col c-1
            if c > 0:
                cc = c - 1
                if not ((r > 0 and ws[r - 1][cc] == WALL_V) or (r < WALL_GRID and ws[r][cc] == WALL_V)):
                    n = cell - 1
                    if not visited[n]:
                        visited[n] = 1
                        queue.append(n)
            # right (r, c+1): blocked by a vertical wall on slot-col c
            if c < BOARD_SIZE - 1:
                if not ((r > 0 and ws[r - 1][c] == WALL_V) or (r < WALL_GRID and ws[r][c] == WALL_V)):
                    n = cell + 1
                    if not visited[n]:
                        visited[n] = 1
                        queue.append(n)
        return False

    def _path_edges(self, player: int):
        """Return the edge set of *one* concrete pawn->goal path, or None if no
        path exists. Edges are canonical (lo, hi) pairs of cell indices.

        This is the cached "current path" used to prune wall-legality checks:
        a candidate wall can only break this player's connectivity if it blocks
        an edge that lies on this path.
        """
        ws = self.wall_slots
        goal_row = self.goal_row(player)
        sr, sc = self.pawns[player]
        start = sr * BOARD_SIZE + sc
        parent = [-2] * (BOARD_SIZE * BOARD_SIZE)  # -2 = unvisited
        parent[start] = -1
        queue = [start]
        qi = 0
        goal_cell = -1
        while qi < len(queue):
            cell = queue[qi]
            qi += 1
            r, c = divmod(cell, BOARD_SIZE)
            if r == goal_row:
                goal_cell = cell
                break
            if r > 0:
                rr = r - 1
                if not ((c > 0 and ws[rr][c - 1] == WALL_H) or (c < WALL_GRID and ws[rr][c] == WALL_H)):
                    n = cell - BOARD_SIZE
                    if parent[n] == -2:
                        parent[n] = cell
                        queue.append(n)
            if r < BOARD_SIZE - 1:
                if not ((c > 0 and ws[r][c - 1] == WALL_H) or (c < WALL_GRID and ws[r][c] == WALL_H)):
                    n = cell + BOARD_SIZE
                    if parent[n] == -2:
                        parent[n] = cell
                        queue.append(n)
            if c > 0:
                cc = c - 1
                if not ((r > 0 and ws[r - 1][cc] == WALL_V) or (r < WALL_GRID and ws[r][cc] == WALL_V)):
                    n = cell - 1
                    if parent[n] == -2:
                        parent[n] = cell
                        queue.append(n)
            if c < BOARD_SIZE - 1:
                if not ((r > 0 and ws[r - 1][c] == WALL_V) or (r < WALL_GRID and ws[r][c] == WALL_V)):
                    n = cell + 1
                    if parent[n] == -2:
                        parent[n] = cell
                        queue.append(n)
        if goal_cell == -1:
            return None
        edges = set()
        cur = goal_cell
        while parent[cur] != -1:
            p = parent[cur]
            edges.add((p, cur) if p < cur else (cur, p))
            cur = p
        return edges

    @staticmethod
    def _wall_blocked_edges(r: int, c: int, orientation: int):
        """The two grid edges (canonical cell-index pairs) a wall would block."""
        if orientation == WALL_H:
            a = r * BOARD_SIZE + c
            a2 = a + 1
            return (a, a + BOARD_SIZE), (a2, a2 + BOARD_SIZE)
        a = r * BOARD_SIZE + c
        b = (r + 1) * BOARD_SIZE + c
        return (a, a + 1), (b, b + 1)

    def legal_wall_slots(self, player: int | None = None) -> list:
        """All (r, c, orientation) wall placements legal for the given player.

        Fast path (cached shortest path): compute one current pawn->goal path
        per player up front. A candidate wall can only sever a player's
        connectivity if it blocks an edge on that player's stored path; if it
        misses both stored paths, those exact paths survive and the wall is
        legal with no BFS. Only candidates that cut a stored path pay for a
        confirming path-existence check, and only for the player(s) affected.
        """
        if player is None:
            player = self.current_player
        if self.walls_left[player] <= 0:
            return []
        # These hold by the game invariant (every reached state is connected).
        sp0 = self._path_edges(0) or set()
        sp1 = self._path_edges(1) or set()
        legal = []
        for r in range(WALL_GRID):
            for c in range(WALL_GRID):
                if self.wall_slots[r][c] != EMPTY:
                    continue
                for orientation in (WALL_H, WALL_V):
                    e1, e2 = self._wall_blocked_edges(r, c, orientation)
                    touches0 = e1 in sp0 or e2 in sp0
                    touches1 = e1 in sp1 or e2 in sp1
                    if not touches0 and not touches1:
                        # Neither stored path is cut -> both paths still exist.
                        legal.append((r, c, orientation))
                        continue
                    self.wall_slots[r][c] = orientation
                    ok = (not touches0 or self._path_exists(0)) and (
                        not touches1 or self._path_exists(1)
                    )
                    self.wall_slots[r][c] = EMPTY
                    if ok:
                        legal.append((r, c, orientation))
        return legal

    def place_wall(self, player: int, r: int, c: int, orientation: int) -> None:
        self.wall_slots[r][c] = orientation
        self.walls_left[player] -= 1
        self.current_player = 1 - self.current_player
