// Package mcts is a Go port of the Python PUCT MCTS, including leaf batching
// with virtual loss (corrected sign) so many leaves evaluate in one forward.
package mcts

import (
	"math"
	"math/rand"

	"github.com/andreiconst/quoridor/engine"
)

// Evaluation is a masked+normalized policy over real actions plus a value from
// the side-to-move's perspective.
type Evaluation struct {
	Policy []float32 // length engine.ActionSize; zero for illegal actions
	Value  float32
}

// Evaluator maps a batch of states to evaluations (one per state).
type Evaluator interface {
	Evaluate(states []*engine.State) []Evaluation
}

// Node is a search-tree node.
type Node struct {
	State      *engine.State
	Prior      float64
	Children   map[int]*Node
	VisitCount float64
	ValueSum   float64
	Player     int
}

func (n *Node) value() float64 {
	if n.VisitCount == 0 {
		return 0
	}
	return n.ValueSum / n.VisitCount
}

func (n *Node) expanded() bool { return len(n.Children) > 0 }

// MCTS holds search configuration.
type MCTS struct {
	Eval           Evaluator
	NumSimulations int
	BatchSize      int
	VirtualLoss    float64
	CPuct          float64
	DirichletAlpha float64
	DirichletEps   float64
	Rng            *rand.Rand
}

// New returns an MCTS with AlphaZero-ish defaults.
func New(eval Evaluator, numSimulations, batchSize int, rng *rand.Rand) *MCTS {
	return &MCTS{
		Eval:           eval,
		NumSimulations: numSimulations,
		BatchSize:      batchSize,
		VirtualLoss:    1.0,
		CPuct:          1.5,
		DirichletAlpha: 0.3,
		DirichletEps:   0.25,
		Rng:            rng,
	}
}

// Run executes the search and returns the populated root.
func (m *MCTS) Run(rootState *engine.State, addNoise bool) *Node {
	root := &Node{State: rootState.Clone(), Player: rootState.Current, Children: map[int]*Node{}}
	rootEval := m.Eval.Evaluate([]*engine.State{root.State})[0]
	m.expand(root, rootEval)
	if addNoise {
		m.addDirichletNoise(root)
	}
	if !root.expanded() {
		return root
	}

	simsDone := 0
	for simsDone < m.NumSimulations {
		leaves, paths, term := m.collectLeaves(root, m.NumSimulations-simsDone)
		simsDone += term
		if len(leaves) == 0 {
			continue
		}
		states := make([]*engine.State, len(leaves))
		for i, lf := range leaves {
			states[i] = lf.State
		}
		results := m.Eval.Evaluate(states)
		for i := range leaves {
			m.revertVirtualLoss(paths[i])
			m.expand(leaves[i], results[i])
			m.backprop(paths[i], float64(results[i].Value))
			simsDone++
		}
	}
	return root
}

func (m *MCTS) collectLeaves(root *Node, remaining int) ([]*Node, [][]*Node, int) {
	var leaves []*Node
	var paths [][]*Node
	pending := map[*Node]bool{}
	terminal := 0
	target := m.BatchSize
	if remaining < target {
		target = remaining
	}
	for len(leaves) < target {
		node := root
		path := []*Node{node}
		for node.expanded() && !node.State.IsTerminal() {
			_, node = m.selectChild(node)
			path = append(path, node)
		}
		leaf := path[len(path)-1]
		if leaf.State.IsTerminal() {
			value := -1.0
			if leaf.State.Winner == leaf.Player {
				value = 1.0
			}
			m.backprop(path, value)
			terminal++
			if terminal >= remaining {
				break
			}
			continue
		}
		if pending[leaf] {
			break // collision: evaluate what we have
		}
		m.applyVirtualLoss(path)
		pending[leaf] = true
		leaves = append(leaves, leaf)
		paths = append(paths, path)
	}
	return leaves, paths, terminal
}

func (m *MCTS) selectChild(node *Node) (int, *Node) {
	best := math.Inf(-1)
	var bestA int
	var bestC *Node
	sqrtTotal := math.Sqrt(math.Max(node.VisitCount, 1))
	for a, child := range node.Children {
		q := -child.value()
		u := m.CPuct * child.Prior * sqrtTotal / (1 + child.VisitCount)
		if score := q + u; score > best {
			best, bestA, bestC = score, a, child
		}
	}
	return bestA, bestC
}

func (m *MCTS) expand(node *Node, e Evaluation) {
	if node.Children == nil {
		node.Children = map[int]*Node{}
	}
	// Masked policy is zero for illegal actions, so nonzero entries are exactly
	// the legal actions (matches the Python expand's legal set).
	for a, p := range e.Policy {
		if p > 0 {
			child := node.State.ApplyAction(a)
			node.Children[a] = &Node{State: child, Prior: float64(p), Player: child.Current}
		}
	}
}

func (m *MCTS) backprop(path []*Node, leafValue float64) {
	value := leafValue
	for i := len(path) - 1; i >= 0; i-- {
		path[i].VisitCount++
		path[i].ValueSum += value
		value = -value
	}
}

// applyVirtualLoss discourages re-walking a path within a batch. Selection uses
// q = -child.value(), so to make a node less attractive to its parent we push
// its value UP (good for the node's own mover) and add a virtual visit.
func (m *MCTS) applyVirtualLoss(path []*Node) {
	for _, n := range path {
		n.VisitCount += m.VirtualLoss
		n.ValueSum += m.VirtualLoss
	}
}

func (m *MCTS) revertVirtualLoss(path []*Node) {
	for _, n := range path {
		n.VisitCount -= m.VirtualLoss
		n.ValueSum -= m.VirtualLoss
	}
}

func (m *MCTS) addDirichletNoise(root *Node) {
	actions := make([]int, 0, len(root.Children))
	for a := range root.Children {
		actions = append(actions, a)
	}
	if len(actions) == 0 {
		return
	}
	noise := dirichlet(m.Rng, m.DirichletAlpha, len(actions))
	for i, a := range actions {
		c := root.Children[a]
		c.Prior = (1-m.DirichletEps)*c.Prior + m.DirichletEps*noise[i]
	}
}

// PolicyFromVisits returns a length-ActionSize distribution from visit counts.
func PolicyFromVisits(root *Node, temperature float64) []float32 {
	policy := make([]float32, engine.ActionSize)
	if temperature <= 1e-3 {
		bestA, bestV := -1, -1.0
		for a, c := range root.Children {
			if c.VisitCount > bestV {
				bestV, bestA = c.VisitCount, a
			}
		}
		if bestA >= 0 {
			policy[bestA] = 1
		}
		return policy
	}
	var sum float64
	weights := map[int]float64{}
	for a, c := range root.Children {
		w := math.Pow(c.VisitCount, 1.0/temperature)
		weights[a] = w
		sum += w
	}
	if sum > 0 {
		for a, w := range weights {
			policy[a] = float32(w / sum)
		}
	}
	return policy
}

// --- Dirichlet via gamma sampling ---

func dirichlet(rng *rand.Rand, alpha float64, n int) []float64 {
	out := make([]float64, n)
	var sum float64
	for i := range out {
		out[i] = gammaSample(rng, alpha)
		sum += out[i]
	}
	if sum > 0 {
		for i := range out {
			out[i] /= sum
		}
	}
	return out
}

// gammaSample draws from Gamma(alpha, 1) (Marsaglia-Tsang, with boost for a<1).
func gammaSample(rng *rand.Rand, alpha float64) float64 {
	if alpha < 1 {
		u := rng.Float64()
		return gammaSample(rng, alpha+1) * math.Pow(u, 1.0/alpha)
	}
	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}
