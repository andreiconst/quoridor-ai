package engine

const (
	NumCells     = BoardSize * BoardSize     // 81
	NumWallSlots = WallGrid * WallGrid       // 64
	ActionSize   = NumCells + 2*NumWallSlots // 209
	Planes       = 6
	PlaneSize    = Planes * BoardSize * BoardSize // 486
)

// PawnAction encodes a pawn destination as an action index.
func PawnAction(r, c int) int { return r*BoardSize + c }

// WallAction encodes a wall placement as an action index.
func WallAction(r, c, orientation int) int {
	base := NumCells
	if orientation == WallV {
		base = NumCells + NumWallSlots
	}
	return base + r*WallGrid + c
}

// DecodeAction returns whether the action is a wall, and its coordinates.
// For pawns: (false, r, c, 0). For walls: (true, r, c, orientation).
func DecodeAction(action int) (isWall bool, r, c, orientation int) {
	if action < NumCells {
		return false, action / BoardSize, action % BoardSize, 0
	}
	action -= NumCells
	if action < NumWallSlots {
		return true, action / WallGrid, action % WallGrid, WallH
	}
	action -= NumWallSlots
	return true, action / WallGrid, action % WallGrid, WallV
}

// LegalActionMask returns a length-209 float32 mask of legal actions.
func (s *State) LegalActionMask() []float32 {
	mask := make([]float32, ActionSize)
	for _, d := range s.LegalPawnDestinations(s.Current) {
		mask[PawnAction(d.R, d.C)] = 1
	}
	for _, w := range s.LegalWallSlots(s.Current) {
		mask[WallAction(w.R, w.C, w.O)] = 1
	}
	return mask
}

// ApplyAction returns a new state with the action applied by the side to move.
func (s *State) ApplyAction(action int) *State {
	ns := s.Clone()
	player := ns.Current
	isWall, r, c, o := DecodeAction(action)
	if isWall {
		ns.PlaceWall(player, r, c, o)
	} else {
		ns.MovePawn(player, Cell{r, c})
	}
	return ns
}

// EncodeState returns the 6x9x9 planes (flattened C-order, length 486) from the
// side-to-move's perspective, matching Python encode_state exactly.
func (s *State) EncodeState() []float32 {
	player := s.Current
	opp := 1 - player
	flip := player == 1

	maybeFlip := func(r, c int) (int, int) {
		if flip {
			return BoardSize - 1 - r, c
		}
		return r, c
	}
	at := func(plane, r, c int) int { return plane*BoardSize*BoardSize + r*BoardSize + c }

	planes := make([]float32, PlaneSize)
	pr, pc := maybeFlip(s.Pawns[player].R, s.Pawns[player].C)
	planes[at(0, pr, pc)] = 1
	or, oc := maybeFlip(s.Pawns[opp].R, s.Pawns[opp].C)
	planes[at(1, or, oc)] = 1

	for r := 0; r < WallGrid; r++ {
		for c := 0; c < WallGrid; c++ {
			switch s.WallSlots[r][c] {
			case WallH:
				fr, fc := maybeFlip(r, c)
				planes[at(2, fr, fc)] = 1
			case WallV:
				fr, fc := maybeFlip(r, c)
				planes[at(3, fr, fc)] = 1
			}
		}
	}

	ownLeft := float32(s.WallsLeft[player]) / 10.0
	oppLeft := float32(s.WallsLeft[opp]) / 10.0
	for i := 0; i < BoardSize*BoardSize; i++ {
		planes[4*BoardSize*BoardSize+i] = ownLeft
		planes[5*BoardSize*BoardSize+i] = oppLeft
	}
	return planes
}

// EncodeActionForPlayer maps an action from canonical (flipped) space back to
// the real board, accounting for the perspective flip in EncodeState.
func (s *State) EncodeActionForPlayer(action int) int {
	if s.Current == 0 {
		return action
	}
	isWall, r, c, o := DecodeAction(action)
	if !isWall {
		return PawnAction(BoardSize-1-r, c)
	}
	return WallAction(WallGrid-1-r, c, o)
}
