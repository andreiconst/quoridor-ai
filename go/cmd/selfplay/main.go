// Command selfplay runs Go MCTS self-play against the Python inference server.
//
//	go run ./cmd/selfplay --socket /tmp/quoridor_infer.sock \
//	    --games 64 --concurrency 16 --sims 100 --batch 16
package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/andreiconst/quoridor/data"
	"github.com/andreiconst/quoridor/selfplay"
	"github.com/andreiconst/quoridor/serving"
)

func main() {
	socket := flag.String("socket", "/tmp/quoridor_infer.sock", "inference server unix socket")
	games := flag.Int("games", 32, "number of self-play games")
	concurrency := flag.Int("concurrency", 16, "concurrent games (goroutines)")
	sims := flag.Int("sims", 100, "MCTS simulations per move")
	batch := flag.Int("batch", 16, "within-game leaf batch size")
	maxBatch := flag.Int("max-batch", 512, "max positions per server forward")
	lingerUs := flag.Int("linger-us", 500, "batcher linger window in microseconds")
	dataDir := flag.String("data-dir", "", "if set, write self-play shards here")
	shardSize := flag.Int("shard-size", 50000, "examples per shard")
	seed := flag.Int64("seed", 1, "rng seed")
	flag.Parse()

	client, err := serving.Dial(*socket)
	if err != nil {
		log.Fatalf("dial %s: %v", *socket, err)
	}
	defer client.Close()
	batcher := serving.NewBatcher(client, *maxBatch, time.Duration(*lingerUs)*time.Microsecond)
	defer batcher.Close()

	var writer *data.ShardWriter
	if *dataDir != "" {
		writer, err = data.NewShardWriter(*dataDir, *shardSize)
		if err != nil {
			log.Fatalf("shard writer: %v", err)
		}
	}

	start := time.Now()
	total, tally := selfplay.GenerateGames(*games, *concurrency, *sims, *batch, batcher, writer, *seed)
	dt := time.Since(start)
	if writer != nil {
		if err := writer.Close(); err != nil {
			log.Fatalf("shard flush: %v", err)
		}
		fmt.Printf("wrote %d examples to %s\n", writer.Total, *dataDir)
	}

	fmt.Printf("games=%d  P0=%d P1=%d draw=%d  examples=%d\n", *games, tally[0], tally[1], tally[2], total)
	fmt.Printf("elapsed=%.1fs  -> %.2f games/s  (%d concurrent, %d sims)\n",
		dt.Seconds(), float64(*games)/dt.Seconds(), *concurrency, *sims)
}
