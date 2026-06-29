package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type goldenCase struct {
	Pawns         [2][2]int               `json:"pawns"`
	WallSlots     [WallGrid][WallGrid]int `json:"wall_slots"`
	WallsLeft     [2]int                  `json:"walls_left"`
	CurrentPlayer int                     `json:"current_player"`
	Planes        [][][]float32           `json:"planes"`
	Mask          []float32               `json:"mask"`
}

func stateFromGolden(c goldenCase) *State {
	return &State{
		Pawns:     [2]Cell{{c.Pawns[0][0], c.Pawns[0][1]}, {c.Pawns[1][0], c.Pawns[1][1]}},
		WallsLeft: c.WallsLeft,
		WallSlots: c.WallSlots,
		Current:   c.CurrentPlayer,
		Winner:    -1,
	}
}

// TestGoldenParity is the cross-language gate: the Go engine must reproduce the
// Python encode_state planes and legal-action masks bit-for-bit.
func TestGoldenParity(t *testing.T) {
	path := filepath.Join("..", "..", "golden", "golden.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("golden file missing (%v); run scripts/dump_golden.py", err)
	}
	var cases []goldenCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("parse golden.json: %v", err)
	}
	if len(cases) < 50 {
		t.Fatalf("expected >50 golden cases, got %d", len(cases))
	}

	for i, c := range cases {
		st := stateFromGolden(c)

		planes := st.EncodeState()
		for p := 0; p < Planes; p++ {
			for r := 0; r < BoardSize; r++ {
				for col := 0; col < BoardSize; col++ {
					got := planes[p*BoardSize*BoardSize+r*BoardSize+col]
					want := c.Planes[p][r][col]
					if got != want {
						t.Fatalf("case %d planes[%d][%d][%d]: got %v want %v", i, p, r, col, got, want)
					}
				}
			}
		}

		mask := st.LegalActionMask()
		for a := 0; a < ActionSize; a++ {
			if mask[a] != c.Mask[a] {
				t.Fatalf("case %d mask[%d]: got %v want %v", i, a, mask[a], c.Mask[a])
			}
		}
	}
	t.Logf("validated %d golden cases", len(cases))
}
