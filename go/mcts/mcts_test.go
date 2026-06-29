package mcts_test

import (
	"math/rand"
	"testing"

	"github.com/andreiconst/quoridor/engine"
	"github.com/andreiconst/quoridor/mcts"
)

// mockEval returns a uniform policy over legal actions and value 0, and records
// the largest batch it was asked to evaluate.
type mockEval struct{ maxBatch int }

func (e *mockEval) Evaluate(states []*engine.State) []mcts.Evaluation {
	if len(states) > e.maxBatch {
		e.maxBatch = len(states)
	}
	out := make([]mcts.Evaluation, len(states))
	for i, s := range states {
		mask := s.LegalActionMask()
		var n float32
		for _, v := range mask {
			n += v
		}
		pol := make([]float32, engine.ActionSize)
		for a, v := range mask {
			if v != 0 {
				pol[a] = v / n
			}
		}
		out[i] = mcts.Evaluation{Policy: pol, Value: 0}
	}
	return out
}

func TestVisitConservation(t *testing.T) {
	for _, bs := range []int{1, 8, 16, 32} {
		eval := &mockEval{}
		m := mcts.New(eval, 200, bs, rand.New(rand.NewSource(1)))
		root := m.Run(engine.Initial(), false)
		var sum float64
		for _, c := range root.Children {
			sum += c.VisitCount
		}
		if sum != 200 {
			t.Fatalf("batch=%d: sum child visits = %v, want 200", bs, sum)
		}
	}
}

// TestWithinGameBatching guards the virtual-loss sign: leaf collection must
// gather more than one distinct leaf per forward.
func TestWithinGameBatching(t *testing.T) {
	eval := &mockEval{}
	m := mcts.New(eval, 64, 16, rand.New(rand.NewSource(1)))
	m.Run(engine.Initial(), false)
	if eval.maxBatch <= 1 {
		t.Fatalf("within-game batch collapsed to %d (virtual-loss sign bug)", eval.maxBatch)
	}
	t.Logf("max within-game leaf batch = %d", eval.maxBatch)
}

func TestRunPicksLegalMove(t *testing.T) {
	eval := &mockEval{}
	m := mcts.New(eval, 50, 8, rand.New(rand.NewSource(2)))
	root := m.Run(engine.Initial(), true)
	policy := mcts.PolicyFromVisits(root, 0)
	best := -1
	for a, p := range policy {
		if p > 0 {
			best = a
		}
	}
	if best < 0 {
		t.Fatal("no move selected")
	}
	if _, ok := root.Children[best]; !ok {
		t.Fatal("selected action is not a child")
	}
}
