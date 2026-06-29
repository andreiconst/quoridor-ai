# Implementation Plan — Go self-play + Python inference

Goal: lift the single-core tree-walk ceiling measured in the asyncio version by
moving self-play (MCTS + game engine) to Go (true-parallel goroutines) while
keeping NN inference and training in Python (one `QuoridorNet`, no ONNX export).
See `PROTOCOL.md` for the interface.

Guiding principle (held throughout this project): **measure before optimizing.**
Prototype the simplest transport, validate correctness against an oracle, then
optimize only what's proven slow.

## Phase 0 — Freeze contract + golden vectors  ✅ (this repo)
- `docs/PROTOCOL.md`: the frozen semantic + wire contract.
- `scripts/dump_golden.py`: emit reference `(planes, legal-action mask)` for a
  set of canonical states as `golden/golden_vectors.npz` + a human-readable
  JSON. This is the oracle the Go engine port must match bit-for-bit.

## Phase 1 — Python inference server, tested without Go  ✅ (this repo)
- `quoridor/serving/infer_server.py`: serve the request/response frames over a
  Unix domain socket; load `QuoridorNet` from a checkpoint; hot-reload on file
  change.
- `quoridor/serving/client.py`: a Python client implementing the same framing —
  doubles as the "fake Go" client to validate the protocol end-to-end in one
  language, and as the reference for the eventual Go client.
- Test: client results must equal a direct in-process `network.predict`.

## Phase 2 — Go engine port
- Port `state.py` (rules + shortest-path wall-legality) and `game.py`
  (`encode_state`, action layout) to Go.
- Port unit tests + the brute-force wall-legality differential test.
- Assert Go `encode_state`/masks match `golden/golden_vectors.npz` exactly.

## Phase 3 — Go MCTS + transport client
- Port PUCT MCTS with virtual loss (corrected sign), leaf batching, goroutine
  games, batcher + collector, free-slot pool.
- Connect to the Phase-1 server (socket first).

## Phase 4 — Data + weight loop
- Go writes the existing shard format (or a simple binary + a tiny Python
  adapter). Python learner trains from shards, writes `current.pt`; server
  hot-reloads. End-to-end self-play → train → improve.

## Phase 5 — Validate + benchmark
- Differential-check Go vs Python self-play on shared seeds.
- Benchmark games/s; confirm parallel Go tree-walks saturate the GPU (the thing
  the asyncio version could not).

## Transport decision
Start on a Unix domain socket (Transport A). Swap to shared memory (Transport B)
only if the per-batch copy is measured to be a real bottleneck.

## Top risks
1. Go engine diverging from Python → silent strength loss. Mitigation: golden
   vectors + differential tests (Phases 0/2).
2. Shared-memory lifecycle/cleanup → defer via socket-first.
3. Engine maintained in two languages → the contract + golden vectors are the
   enforced bridge; re-dump and re-validate on any encoding change.
