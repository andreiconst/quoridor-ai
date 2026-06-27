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
        start = self.pawns[player]
        goal_row = self.goal_row(player)
        visited = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if r == goal_row:
                return True
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (
                    self._in_bounds(nr, nc)
                    and (nr, nc) not in visited
                    and not self._edge_blocked(r, c, nr, nc)
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def legal_wall_slots(self, player: int | None = None) -> list:
        """All (r, c, orientation) wall placements legal for the given player."""
        if player is None:
            player = self.current_player
        if self.walls_left[player] <= 0:
            return []
        legal = []
        for r in range(WALL_GRID):
            for c in range(WALL_GRID):
                if self.wall_slots[r][c] != EMPTY:
                    continue
                for orientation in (WALL_H, WALL_V):
                    self.wall_slots[r][c] = orientation
                    if self._path_exists(0) and self._path_exists(1):
                        legal.append((r, c, orientation))
                    self.wall_slots[r][c] = EMPTY
        return legal

    def place_wall(self, player: int, r: int, c: int, orientation: int) -> None:
        self.wall_slots[r][c] = orientation
        self.walls_left[player] -= 1
        self.current_player = 1 - self.current_player
