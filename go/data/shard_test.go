package data

import (
	"path/filepath"
	"testing"
)

func TestShardRoundTrip(t *testing.T) {
	dir := t.TempDir()
	w, err := NewShardWriter(dir, 3) // small shard size to force multiple files
	if err != nil {
		t.Fatal(err)
	}
	n := 7
	for i := 0; i < n; i++ {
		planes := make([]float32, PlaneSize)
		policy := make([]float32, ActionSz)
		planes[i] = float32(i) + 0.5
		policy[i] = 1.0
		if err := w.Add(planes, policy, float32(i)-3); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	shards, _ := filepath.Glob(filepath.Join(dir, "go_*.qsh"))
	if len(shards) != 3 { // 3,3,1
		t.Fatalf("expected 3 shards, got %d", len(shards))
	}

	total := 0
	for _, p := range shards {
		s, err := ReadShard(p)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < s.Count; i++ {
			global := total + i
			if got := s.Planes[i*PlaneSize+global]; got != float32(global)+0.5 {
				t.Fatalf("plane[%d] = %v", global, got)
			}
			if got := s.Policies[i*ActionSz+global]; got != 1.0 {
				t.Fatalf("policy[%d] = %v", global, got)
			}
			if got := s.Values[i]; got != float32(global)-3 {
				t.Fatalf("value[%d] = %v", global, got)
			}
		}
		total += s.Count
	}
	if total != n {
		t.Fatalf("read %d examples, wrote %d", total, n)
	}
}
