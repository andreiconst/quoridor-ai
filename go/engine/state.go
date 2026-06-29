// Package engine is a Go port of the Python Quoridor rules engine
// (quoridor/engine/state.py + game.py). It must match the Python encode_state
// and action layout bit-for-bit; see docs/PROTOCOL.md and the golden tests.
package engine

const (
	BoardSize      = 9
	WallGrid       = BoardSize - 1 // 8
	WallsPerPlayer = 10

	Empty = 0
	WallH = 1
	WallV = 2
)

var dirs = [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}

// Cell is a (row, col) board coordinate.
type Cell struct{ R, C int }

// Wall is a (row, col, orientation) wall-slot placement.
type Wall struct {
	R, C, O int
}

// State is a Quoridor position. Winner is -1 when the game is ongoing.
type State struct {
	Pawns     [2]Cell
	WallsLeft [2]int
	WallSlots [WallGrid][WallGrid]int
	Current   int
	Winner    int
}

// Initial returns the standard start position.
func Initial() *State {
	return &State{
		Pawns:     [2]Cell{{0, BoardSize / 2}, {BoardSize - 1, BoardSize / 2}},
		WallsLeft: [2]int{WallsPerPlayer, WallsPerPlayer},
		Current:   0,
		Winner:    -1,
	}
}

// Clone returns a deep copy.
func (s *State) Clone() *State {
	c := *s // arrays copy by value
	return &c
}

func (s *State) goalRow(player int) int {
	if player == 0 {
		return BoardSize - 1
	}
	return 0
}

// IsTerminal reports whether the game has been won.
func (s *State) IsTerminal() bool { return s.Winner != -1 }

func inBounds(r, c int) bool { return r >= 0 && r < BoardSize && c >= 0 && c < BoardSize }

// edgeBlocked reports whether a wall blocks the edge between two adjacent cells.
func (s *State) edgeBlocked(r1, c1, r2, c2 int) bool {
	if r1 == r2 { // horizontal move (left/right): vertical walls block it
		c := min(c1, c2)
		for _, rr := range [2]int{r1 - 1, r1} {
			if rr >= 0 && rr < WallGrid && c >= 0 && c < WallGrid && s.WallSlots[rr][c] == WallV {
				return true
			}
		}
		return false
	}
	// vertical move (up/down): horizontal walls block it
	r := min(r1, r2)
	for _, cc := range [2]int{c1 - 1, c1} {
		if r >= 0 && r < WallGrid && cc >= 0 && cc < WallGrid && s.WallSlots[r][cc] == WallH {
			return true
		}
	}
	return false
}

// LegalPawnDestinations returns every cell the player's pawn may move to,
// including straight and diagonal jumps over the opponent.
func (s *State) LegalPawnDestinations(player int) []Cell {
	r, c := s.Pawns[player].R, s.Pawns[player].C
	opp := s.Pawns[1-player]
	var dests []Cell
	for _, d := range dirs {
		dr, dc := d[0], d[1]
		nr, nc := r+dr, c+dc
		if !inBounds(nr, nc) || s.edgeBlocked(r, c, nr, nc) {
			continue
		}
		if nr == opp.R && nc == opp.C {
			jr, jc := nr+dr, nc+dc
			if inBounds(jr, jc) && !s.edgeBlocked(nr, nc, jr, jc) {
				dests = append(dests, Cell{jr, jc}) // straight jump
			} else {
				for _, sd := range dirs { // diagonal jumps (perpendicular dirs)
					if (sd[0] == dr && sd[1] == dc) || (sd[0] == -dr && sd[1] == -dc) {
						continue
					}
					sr, sc := nr+sd[0], nc+sd[1]
					if inBounds(sr, sc) && !s.edgeBlocked(nr, nc, sr, sc) {
						dests = append(dests, Cell{sr, sc})
					}
				}
			}
		} else {
			dests = append(dests, Cell{nr, nc})
		}
	}
	return dests
}

// MovePawn moves the player's pawn and flips the side to move.
func (s *State) MovePawn(player int, dest Cell) {
	s.Pawns[player] = dest
	if dest.R == s.goalRow(player) {
		s.Winner = player
	}
	s.Current = 1 - s.Current
}

// PlaceWall places a wall and flips the side to move.
func (s *State) PlaceWall(player, r, c, orientation int) {
	s.WallSlots[r][c] = orientation
	s.WallsLeft[player]--
	s.Current = 1 - s.Current
}

// pathExists runs a BFS (inlined edge checks) from the pawn to its goal row.
func (s *State) pathExists(player int) bool {
	goal := s.goalRow(player)
	start := s.Pawns[player].R*BoardSize + s.Pawns[player].C
	var visited [BoardSize * BoardSize]bool
	visited[start] = true
	queue := []int{start}
	for qi := 0; qi < len(queue); qi++ {
		cell := queue[qi]
		r, c := cell/BoardSize, cell%BoardSize
		if r == goal {
			return true
		}
		if r > 0 {
			rr := r - 1
			if !((c > 0 && s.WallSlots[rr][c-1] == WallH) || (c < WallGrid && s.WallSlots[rr][c] == WallH)) {
				if n := cell - BoardSize; !visited[n] {
					visited[n] = true
					queue = append(queue, n)
				}
			}
		}
		if r < BoardSize-1 {
			if !((c > 0 && s.WallSlots[r][c-1] == WallH) || (c < WallGrid && s.WallSlots[r][c] == WallH)) {
				if n := cell + BoardSize; !visited[n] {
					visited[n] = true
					queue = append(queue, n)
				}
			}
		}
		if c > 0 {
			cc := c - 1
			if !((r > 0 && s.WallSlots[r-1][cc] == WallV) || (r < WallGrid && s.WallSlots[r][cc] == WallV)) {
				if n := cell - 1; !visited[n] {
					visited[n] = true
					queue = append(queue, n)
				}
			}
		}
		if c < BoardSize-1 {
			if !((r > 0 && s.WallSlots[r-1][c] == WallV) || (r < WallGrid && s.WallSlots[r][c] == WallV)) {
				if n := cell + 1; !visited[n] {
					visited[n] = true
					queue = append(queue, n)
				}
			}
		}
	}
	return false
}

func edgeKey(a, b int) int {
	if a < b {
		return a*BoardSize*BoardSize + b
	}
	return b*BoardSize*BoardSize + a
}

// pathEdges returns the edge set of one concrete pawn->goal path, or nil if no
// path exists. Edges are canonical keys of cell-index pairs.
func (s *State) pathEdges(player int) map[int]bool {
	goal := s.goalRow(player)
	start := s.Pawns[player].R*BoardSize + s.Pawns[player].C
	var parent [BoardSize * BoardSize]int
	for i := range parent {
		parent[i] = -2
	}
	parent[start] = -1
	queue := []int{start}
	goalCell := -1
	for qi := 0; qi < len(queue) && goalCell == -1; qi++ {
		cell := queue[qi]
		r, c := cell/BoardSize, cell%BoardSize
		if r == goal {
			goalCell = cell
			break
		}
		try := func(n int) {
			if parent[n] == -2 {
				parent[n] = cell
				queue = append(queue, n)
			}
		}
		if r > 0 {
			rr := r - 1
			if !((c > 0 && s.WallSlots[rr][c-1] == WallH) || (c < WallGrid && s.WallSlots[rr][c] == WallH)) {
				try(cell - BoardSize)
			}
		}
		if r < BoardSize-1 {
			if !((c > 0 && s.WallSlots[r][c-1] == WallH) || (c < WallGrid && s.WallSlots[r][c] == WallH)) {
				try(cell + BoardSize)
			}
		}
		if c > 0 {
			cc := c - 1
			if !((r > 0 && s.WallSlots[r-1][cc] == WallV) || (r < WallGrid && s.WallSlots[r][cc] == WallV)) {
				try(cell - 1)
			}
		}
		if c < BoardSize-1 {
			if !((r > 0 && s.WallSlots[r-1][c] == WallV) || (r < WallGrid && s.WallSlots[r][c] == WallV)) {
				try(cell + 1)
			}
		}
	}
	if goalCell == -1 {
		return nil
	}
	edges := make(map[int]bool)
	cur := goalCell
	for parent[cur] != -1 {
		p := parent[cur]
		edges[edgeKey(p, cur)] = true
		cur = p
	}
	return edges
}

// wallBlockedEdges returns the two grid edges a wall would block.
func wallBlockedEdges(r, c, orientation int) (int, int) {
	if orientation == WallH {
		a := r*BoardSize + c
		a2 := a + 1
		return edgeKey(a, a+BoardSize), edgeKey(a2, a2+BoardSize)
	}
	a := r*BoardSize + c
	b := (r+1)*BoardSize + c
	return edgeKey(a, a+1), edgeKey(b, b+1)
}

// LegalWallSlots returns all legal wall placements for the player, using the
// cached shortest-path prune: a candidate only needs a BFS if it cuts a stored
// path.
func (s *State) LegalWallSlots(player int) []Wall {
	if s.WallsLeft[player] <= 0 {
		return nil
	}
	sp0 := s.pathEdges(0)
	sp1 := s.pathEdges(1)
	var legal []Wall
	for r := 0; r < WallGrid; r++ {
		for c := 0; c < WallGrid; c++ {
			if s.WallSlots[r][c] != Empty {
				continue
			}
			for _, o := range [2]int{WallH, WallV} {
				e1, e2 := wallBlockedEdges(r, c, o)
				t0 := sp0[e1] || sp0[e2]
				t1 := sp1[e1] || sp1[e2]
				if !t0 && !t1 {
					legal = append(legal, Wall{r, c, o})
					continue
				}
				s.WallSlots[r][c] = o
				ok := (!t0 || s.pathExists(0)) && (!t1 || s.pathExists(1))
				s.WallSlots[r][c] = Empty
				if ok {
					legal = append(legal, Wall{r, c, o})
				}
			}
		}
	}
	return legal
}
