package mcts

import "github.com/andreiconst/quoridor/engine"

// Infer runs a raw batched network forward: planes is count*486 float32,
// returns (probs count*209, values count). serving.Client and serving.Batcher
// satisfy this structurally.
type Infer interface {
	Infer(planes []float32, count int) (probs []float32, values []float32, err error)
}

// NetEvaluator encodes states, runs batched inference, and maps the network's
// canonical policy onto each state's real legal actions (the masking + the
// player-1 perspective remap), matching Python _mask_probs.
type NetEvaluator struct {
	infer Infer
}

func NewNetEvaluator(infer Infer) *NetEvaluator { return &NetEvaluator{infer: infer} }

func (e *NetEvaluator) Evaluate(states []*engine.State) []Evaluation {
	count := len(states)
	planes := make([]float32, count*engine.PlaneSize)
	for i, s := range states {
		copy(planes[i*engine.PlaneSize:], s.EncodeState())
	}

	probs, values, err := e.infer.Infer(planes, count)
	if err != nil {
		panic(err) // self-play cannot proceed without inference
	}

	out := make([]Evaluation, count)
	for i, s := range states {
		base := i * engine.ActionSize
		mask := s.LegalActionMask()
		realPolicy := make([]float32, engine.ActionSize)
		var totalP float32
		var legalCount float32
		for a := 0; a < engine.ActionSize; a++ {
			if mask[a] != 0 {
				realPolicy[a] = probs[base+s.EncodeActionForPlayer(a)]
				totalP += realPolicy[a]
				legalCount++
			}
		}
		if totalP > 0 {
			for a := range realPolicy {
				realPolicy[a] /= totalP
			}
		} else if legalCount > 0 {
			for a := 0; a < engine.ActionSize; a++ {
				if mask[a] != 0 {
					realPolicy[a] = 1.0 / legalCount
				}
			}
		}
		out[i] = Evaluation{Policy: realPolicy, Value: values[i]}
	}
	return out
}
