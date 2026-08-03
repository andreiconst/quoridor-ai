// Package data writes self-play examples as simple binary shards that the
// Python learner reads (see quoridor/serving/go_shards.py). Format per file:
//
//	magic u32 | count u32 | planeSize u32 (486) | actionSize u32 (209)
//	then count records: planes[486] f32 | policy[209] f32 | value f32 | searchValue f32
//
// value is the game outcome and searchValue is the MCTS root value at that
// position (both from the side-to-move's perspective); the learner blends them
// into a lower-variance value target. All little-endian; plain float32 keeps
// the cross-language reader trivial.
package data

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const (
	Magic     = 0x51534832 // "QSH2" (bumped: records now carry searchValue)
	PlaneSize = 6 * 9 * 9  // 486
	ActionSz  = 209
)

// ShardWriter accumulates examples and flushes fixed-size binary shards. Safe
// for concurrent use by many self-play goroutines.
type ShardWriter struct {
	dir          string
	shardSize    int
	mu           sync.Mutex
	planes       []float32
	policies     []float32
	values       []float32
	searchValues []float32
	n            int
	idx          int
	stamp        int64 // per-writer unique prefix so pruning by mtime never collides
	Total        int
}

// Shard is a decoded shard (flat arrays).
type Shard struct {
	Count        int
	Planes       []float32 // Count*486
	Policies     []float32 // Count*209
	Values       []float32 // Count
	SearchValues []float32 // Count
}

// ReadShard decodes a .qsh file (used by tests; the learner reads in Python).
func ReadShard(path string) (*Shard, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 16 || binary.LittleEndian.Uint32(raw[0:]) != Magic {
		return nil, fmt.Errorf("bad magic in %s", path)
	}
	count := int(binary.LittleEndian.Uint32(raw[4:]))
	off := 16
	f32 := func() float32 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[off:]))
		off += 4
		return v
	}
	s := &Shard{Count: count,
		Planes:       make([]float32, count*PlaneSize),
		Policies:     make([]float32, count*ActionSz),
		Values:       make([]float32, count),
		SearchValues: make([]float32, count),
	}
	for i := 0; i < count; i++ {
		for j := 0; j < PlaneSize; j++ {
			s.Planes[i*PlaneSize+j] = f32()
		}
		for j := 0; j < ActionSz; j++ {
			s.Policies[i*ActionSz+j] = f32()
		}
		s.Values[i] = f32()
		s.SearchValues[i] = f32()
	}
	return s, nil
}

// NewShardWriter creates dir and continues numbering after any existing shards.
func NewShardWriter(dir string, shardSize int) (*ShardWriter, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	// A per-writer timestamp prefix keeps filenames unique across invocations,
	// so the training loop can prune old shards by mtime without name collisions.
	return &ShardWriter{dir: dir, shardSize: shardSize, stamp: time.Now().UnixNano()}, nil
}

// Add appends one example (planes len 486, policy len 209). value is the game
// outcome and searchValue the MCTS root value, both side-to-move perspective.
// Thread-safe.
func (w *ShardWriter) Add(planes, policy []float32, value, searchValue float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.planes = append(w.planes, planes...)
	w.policies = append(w.policies, policy...)
	w.values = append(w.values, value)
	w.searchValues = append(w.searchValues, searchValue)
	w.n++
	if w.n >= w.shardSize {
		return w.flushLocked()
	}
	return nil
}

// Close flushes any buffered examples.
func (w *ShardWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.flushLocked()
}

func (w *ShardWriter) flushLocked() error {
	if w.n == 0 {
		return nil
	}
	path := filepath.Join(w.dir, fmt.Sprintf("go_%d_%06d.qsh", w.stamp, w.idx))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<20)

	var hdr [16]byte
	binary.LittleEndian.PutUint32(hdr[0:], Magic)
	binary.LittleEndian.PutUint32(hdr[4:], uint32(w.n))
	binary.LittleEndian.PutUint32(hdr[8:], PlaneSize)
	binary.LittleEndian.PutUint32(hdr[12:], ActionSz)
	if _, err := bw.Write(hdr[:]); err != nil {
		f.Close()
		return err
	}

	var scratch [4]byte
	putF32 := func(v float32) error {
		binary.LittleEndian.PutUint32(scratch[:], math.Float32bits(v))
		_, err := bw.Write(scratch[:])
		return err
	}
	for i := 0; i < w.n; i++ {
		for _, v := range w.planes[i*PlaneSize : (i+1)*PlaneSize] {
			if err := putF32(v); err != nil {
				f.Close()
				return err
			}
		}
		for _, v := range w.policies[i*ActionSz : (i+1)*ActionSz] {
			if err := putF32(v); err != nil {
				f.Close()
				return err
			}
		}
		if err := putF32(w.values[i]); err != nil {
			f.Close()
			return err
		}
		if err := putF32(w.searchValues[i]); err != nil {
			f.Close()
			return err
		}
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	w.Total += w.n
	w.idx++
	w.planes = w.planes[:0]
	w.policies = w.policies[:0]
	w.values = w.values[:0]
	w.searchValues = w.searchValues[:0]
	w.n = 0
	return nil
}
