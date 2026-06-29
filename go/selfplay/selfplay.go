// Package selfplay runs Go MCTS games, optionally many concurrently so their
// leaf evaluations batch together through a shared inference path.
package selfplay

import (
	"math/rand"
	"sync"
	"sync/atomic"

	"github.com/andreiconst/quoridor/data"
	"github.com/andreiconst/quoridor/engine"
	"github.com/andreiconst/quoridor/mcts"
)

const (
	MaxMoves         = 200
	TemperatureMoves = 20
)

// Example is one training sample: planes (network input) + canonical policy
// (remapped to the flipped action space, matching encode_state) + outcome from
// the side-to-move's perspective.
type Example struct {
	Planes  []float32
	Policy  []float32
	Player  int
	Outcome float32
}

// PlayOneGame plays a full self-play game and returns its examples + winner.
func PlayOneGame(eval mcts.Evaluator, numSims, batchSize int, rng *rand.Rand) ([]Example, int) {
	state := engine.Initial()
	m := mcts.New(eval, numSims, batchSize, rng)
	var examples []Example

	for ply := 0; ply < MaxMoves; ply++ {
		if !anyLegal(state) {
			break
		}
		root := m.Run(state, true)
		temp := 1.0
		if ply >= TemperatureMoves {
			temp = 0.1
		}
		policy := mcts.PolicyFromVisits(root, temp)

		canon := make([]float32, engine.ActionSize)
		for a, p := range policy {
			if p > 0 {
				canon[state.EncodeActionForPlayer(a)] = p
			}
		}
		examples = append(examples, Example{Planes: state.EncodeState(), Policy: canon, Player: state.Current})

		a := sampleAction(rng, policy)
		state = root.Children[a].State
		if state.IsTerminal() {
			break
		}
	}

	winner := state.Winner
	for i := range examples {
		switch {
		case winner == -1:
			examples[i].Outcome = 0
		case examples[i].Player == winner:
			examples[i].Outcome = 1
		default:
			examples[i].Outcome = -1
		}
	}
	return examples, winner
}

// GenerateGames runs nGames across `concurrency` goroutines, all sharing infer
// (a serving.Batcher) so concurrent tree-walks merge into batched forwards.
// If writer is non-nil, every example is persisted. Returns total examples and
// a [P0, P1, draw] win tally.
func GenerateGames(nGames, concurrency, numSims, batchSize int, infer mcts.Infer, writer *data.ShardWriter, seed int64) (int, [3]int) {
	var remaining = int64(nGames)
	var totalExamples int64
	var tally [3]int64
	var wg sync.WaitGroup

	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed + int64(id)))
			eval := mcts.NewNetEvaluator(infer)
			for atomic.AddInt64(&remaining, -1) >= 0 {
				ex, winner := PlayOneGame(eval, numSims, batchSize, rng)
				atomic.AddInt64(&totalExamples, int64(len(ex)))
				if writer != nil {
					for i := range ex {
						_ = writer.Add(ex[i].Planes, ex[i].Policy, ex[i].Outcome)
					}
				}
				idx := 2
				if winner == 0 {
					idx = 0
				} else if winner == 1 {
					idx = 1
				}
				atomic.AddInt64(&tally[idx], 1)
			}
		}(w)
	}
	wg.Wait()
	return int(totalExamples), [3]int{int(tally[0]), int(tally[1]), int(tally[2])}
}

func anyLegal(s *engine.State) bool {
	for _, v := range s.LegalActionMask() {
		if v != 0 {
			return true
		}
	}
	return false
}

func sampleAction(rng *rand.Rand, policy []float32) int {
	r := rng.Float64()
	var cum float64
	last := -1
	for a, p := range policy {
		if p > 0 {
			last = a
			cum += float64(p)
			if r <= cum {
				return a
			}
		}
	}
	return last
}
