package engine

import (
	"math/rand"
	"sort"
	"testing"
)

func hasCell(cells []Cell, want Cell) bool {
	for _, c := range cells {
		if c == want {
			return true
		}
	}
	return false
}

func TestInitialPawnMoves(t *testing.T) {
	s := Initial()
	dests := s.LegalPawnDestinations(0)
	if !hasCell(dests, Cell{1, 4}) {
		t.Fatalf("expected forward move (1,4) in %v", dests)
	}
	if len(dests) != 3 { // forward, left, right
		t.Fatalf("expected 3 initial moves, got %d: %v", len(dests), dests)
	}
}

func TestStraightJump(t *testing.T) {
	s := Initial()
	s.Pawns = [2]Cell{{3, 4}, {4, 4}}
	s.Current = 0
	dests := s.LegalPawnDestinations(0)
	if !hasCell(dests, Cell{5, 4}) {
		t.Fatalf("expected straight jump (5,4) in %v", dests)
	}
	if hasCell(dests, Cell{4, 4}) {
		t.Fatalf("must not land on opponent (4,4): %v", dests)
	}
}

func TestDiagonalJumpWhenBlocked(t *testing.T) {
	s := Initial()
	s.Pawns = [2]Cell{{3, 4}, {4, 4}}
	s.WallSlots[4][3] = WallH
	s.WallSlots[4][4] = WallH
	s.Current = 0
	dests := s.LegalPawnDestinations(0)
	if hasCell(dests, Cell{5, 4}) {
		t.Fatalf("straight jump should be blocked: %v", dests)
	}
	if !hasCell(dests, Cell{4, 3}) || !hasCell(dests, Cell{4, 5}) {
		t.Fatalf("expected diagonal jumps (4,3),(4,5): %v", dests)
	}
}

func TestWinDetection(t *testing.T) {
	s := Initial()
	s.Pawns = [2]Cell{{7, 4}, {0, 0}}
	s.Current = 0
	s.MovePawn(0, Cell{8, 4})
	if s.Winner != 0 || !s.IsTerminal() {
		t.Fatalf("expected player 0 win, winner=%d", s.Winner)
	}
}

func TestWallDecrement(t *testing.T) {
	s := Initial()
	legal := s.LegalWallSlots(0)
	if len(legal) == 0 {
		t.Fatal("expected legal walls at start")
	}
	w := legal[0]
	ns := s.ApplyAction(WallAction(w.R, w.C, w.O))
	if ns.WallsLeft[0] != 9 {
		t.Fatalf("expected 9 walls left, got %d", ns.WallsLeft[0])
	}
	if ns.WallSlots[w.R][w.C] != w.O {
		t.Fatal("wall not placed")
	}
}

// bruteLegalWalls is the naive reference: full BFS for every candidate.
func bruteLegalWalls(s *State, player int) []Wall {
	if s.WallsLeft[player] <= 0 {
		return nil
	}
	var legal []Wall
	for r := 0; r < WallGrid; r++ {
		for c := 0; c < WallGrid; c++ {
			if s.WallSlots[r][c] != Empty {
				continue
			}
			for _, o := range [2]int{WallH, WallV} {
				s.WallSlots[r][c] = o
				if s.pathExists(0) && s.pathExists(1) {
					legal = append(legal, Wall{r, c, o})
				}
				s.WallSlots[r][c] = Empty
			}
		}
	}
	return legal
}

func sortWalls(w []Wall) {
	sort.Slice(w, func(i, j int) bool {
		if w[i].R != w[j].R {
			return w[i].R < w[j].R
		}
		if w[i].C != w[j].C {
			return w[i].C < w[j].C
		}
		return w[i].O < w[j].O
	})
}

// TestWallLegalityMatchesBruteForce guards the shortest-path prune against a
// naive full-BFS reference across many random *valid* positions (the game
// invariant: both players always have a path). Mirrors the Python test.
func TestWallLegalityMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(12345))
	tested := 0
	for trial := 0; trial < 6000; trial++ {
		s := Initial()
		nwalls := rng.Intn(21)
		for i := 0; i < nwalls; i++ {
			r, c := rng.Intn(WallGrid), rng.Intn(WallGrid)
			o := WallH
			if rng.Intn(2) == 1 {
				o = WallV
			}
			if s.WallSlots[r][c] == Empty {
				s.WallSlots[r][c] = o
				if !(s.pathExists(0) && s.pathExists(1)) {
					s.WallSlots[r][c] = Empty
				}
			}
		}
		s.Pawns = [2]Cell{{rng.Intn(9), rng.Intn(9)}, {rng.Intn(9), rng.Intn(9)}}
		if s.Pawns[0] == s.Pawns[1] {
			continue
		}
		if !(s.pathExists(0) && s.pathExists(1)) {
			continue
		}
		tested++
		for _, pl := range []int{0, 1} {
			got := s.LegalWallSlots(pl)
			want := bruteLegalWalls(s, pl)
			sortWalls(got)
			sortWalls(want)
			if len(got) != len(want) {
				t.Fatalf("trial %d player %d: len got=%d want=%d", trial, pl, len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("trial %d player %d: %v vs %v", trial, pl, got[i], want[i])
				}
			}
		}
	}
	if tested < 2000 {
		t.Fatalf("only %d valid states tested", tested)
	}
	t.Logf("validated wall legality on %d valid states", tested)
}
