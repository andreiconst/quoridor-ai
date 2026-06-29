package serving

import "time"

// Batcher merges Infer calls from many goroutines into single forward passes,
// the whole point of the Go self-play design: parallel tree-walks feed one
// batched GPU inference. Its Infer method is safe for concurrent use and
// structurally satisfies the mcts.Infer interface.
type Batcher struct {
	reqCh    chan *batchReq
	client   *Client
	maxBatch int
	linger   time.Duration
}

type batchReq struct {
	planes []float32
	count  int
	reply  chan batchResp
}

type batchResp struct {
	probs  []float32
	values []float32
	err    error
}

// NewBatcher starts the batching loop. maxBatch caps positions per forward;
// linger is a short window to let concurrent requests accumulate (0 = fire as
// soon as the queue drains).
func NewBatcher(client *Client, maxBatch int, linger time.Duration) *Batcher {
	b := &Batcher{
		reqCh:    make(chan *batchReq, 1024),
		client:   client,
		maxBatch: maxBatch,
		linger:   linger,
	}
	go b.loop()
	return b
}

// Infer blocks until this request's slice of a merged batch is ready.
func (b *Batcher) Infer(planes []float32, count int) ([]float32, []float32, error) {
	r := &batchReq{planes: planes, count: count, reply: make(chan batchResp, 1)}
	b.reqCh <- r
	res := <-r.reply
	return res.probs, res.values, res.err
}

// Close stops the batcher loop after draining.
func (b *Batcher) Close() { close(b.reqCh) }

func (b *Batcher) loop() {
	for {
		first, ok := <-b.reqCh
		if !ok {
			return
		}
		batch := []*batchReq{first}
		total := first.count

		if b.linger > 0 {
			time.Sleep(b.linger) // let concurrent requests pile up
		}

		closed := false
	drain:
		for total < b.maxBatch {
			select {
			case r, ok := <-b.reqCh:
				if !ok {
					closed = true
					break drain
				}
				batch = append(batch, r)
				total += r.count
			default:
				break drain
			}
		}

		b.dispatch(batch, total)
		if closed {
			return
		}
	}
}

func (b *Batcher) dispatch(batch []*batchReq, total int) {
	buf := make([]float32, total*planeSize)
	off := 0
	for _, r := range batch {
		copy(buf[off*planeSize:], r.planes)
		off += r.count
	}

	probs, values, err := b.client.Infer(buf, total)

	off = 0
	for _, r := range batch {
		if err != nil {
			r.reply <- batchResp{err: err}
			continue
		}
		k := r.count
		r.reply <- batchResp{
			probs:  probs[off*actionSize : (off+k)*actionSize],
			values: values[off : off+k],
		}
		off += k
	}
}
