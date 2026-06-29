// Package serving speaks the inference wire protocol (docs/PROTOCOL.md) to the
// Python inference server over a Unix domain socket, and provides a batcher
// that merges concurrent inference calls into single forward passes.
package serving

import (
	"encoding/binary"
	"io"
	"math"
	"net"
)

const (
	planeSize  = 6 * 9 * 9 // 486
	actionSize = 209
)

// Client is a single connection to the inference server. Infer is NOT safe for
// concurrent use; share access via a Batcher.
type Client struct {
	conn net.Conn
}

// Dial connects to the inference server's Unix socket.
func Dial(socketPath string) (*Client, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn}, nil
}

func (c *Client) Close() error { return c.conn.Close() }

// Infer sends a batch of `count` encoded states (planes is count*486 float32)
// and returns (probs count*209, values count).
func (c *Client) Infer(planes []float32, count int) ([]float32, []float32, error) {
	req := make([]byte, 4+len(planes)*4)
	binary.LittleEndian.PutUint32(req[:4], uint32(count))
	for i, v := range planes {
		binary.LittleEndian.PutUint32(req[4+i*4:], math.Float32bits(v))
	}
	if _, err := c.conn.Write(req); err != nil {
		return nil, nil, err
	}

	var hdr [4]byte
	if _, err := io.ReadFull(c.conn, hdr[:]); err != nil {
		return nil, nil, err
	}
	n := int(binary.LittleEndian.Uint32(hdr[:]))

	polBytes := make([]byte, n*actionSize*4)
	if _, err := io.ReadFull(c.conn, polBytes); err != nil {
		return nil, nil, err
	}
	valBytes := make([]byte, n*4)
	if _, err := io.ReadFull(c.conn, valBytes); err != nil {
		return nil, nil, err
	}
	return bytesToF32(polBytes), bytesToF32(valBytes), nil
}

func bytesToF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
