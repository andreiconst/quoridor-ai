# Inference Protocol (Go self-play ↔ Python inference)

This document is the **frozen contract** between a (future) Go self-play process
and the Python inference process. Both sides must agree exactly, or self-play
silently degrades. The semantic contract (what the bytes mean) matters far more
than the transport (how they move).

## Architecture

```
 Go process (self-play)                         Python process (inference)
 ┌───────────────────────────┐                 ┌──────────────────────────┐
 │ N goroutines: MCTS games  │   transport     │ load QuoridorNet         │
 │ batcher: pack leaves ─────┼──── frames ─────┤ forward(batch) on GPU    │
 │ collector: scatter results┼─────────────────┤ write policy+value back  │
 └───────────────────────────┘                 │ reload weights on change │
            │ writes                            └──────────────────────────┘
            ▼ data shards (.npz)                          ▲ reads
        filesystem  ◄───────── checkpoint (.pt) ──────────┘  Python learner
                                                              (trains offline)
```

Two couplings: a hot-path inference channel (this protocol) and the filesystem
(self-play writes `(state, policy, value)` shards; the learner writes
checkpoints that inference hot-reloads). Inference and the learner are both
plain Python sharing the same `QuoridorNet` class — **no model export, no second
architecture definition.**

## 1. Semantic contract (freeze this)

| Item | Value |
|---|---|
| Planes shape | `(count, 6, 9, 9)` float32, C-contiguous (row-major) |
| Plane order | 0 own pawn, 1 opp pawn, 2 H-walls, 3 V-walls, 4 own walls-left/10, 5 opp walls-left/10 |
| Perspective | board flipped vertically when side-to-move is player 1 (matches `quoridor.engine.game.encode_state`) |
| Policy shape | `(count, 209)` float32, **canonical** action space (same flip) |
| Action layout | `0..80` pawn cell `r*9+c`; `81..144` H-wall slot `r*8+c`; `145..208` V-wall slot `r*8+c` |
| Value | `(count,)` float32 in `[-1, 1]`, side-to-move perspective |
| Endianness | little-endian |

The Go engine must reproduce `encode_state` and this action indexing
**bit-for-bit**. The golden vectors in `scripts/dump_golden.py` are the oracle:
the Go port must match them exactly.

Constants (authoritative source: `quoridor/engine/game.py`):
`BOARD_SIZE=9`, `WALL_GRID=8`, `PLANES=6`, `PLANE_SIZE=486`, `ACTION_SIZE=209`.

## 2. Wire protocol (transport-agnostic)

```
Request  frame:  count:u32 │ planes[count*486] f32
Response frame:  count:u32 │ policy[count*209] f32 │ value[count] f32
```

`count` is little-endian uint32. Floats are little-endian float32. The request's
`planes` is the flattened `(count,6,9,9)` array; the response's `policy` is the
flattened `(count,209)` array followed by `count` values.

### Transport A — Unix domain socket (v1, recommended to start)

Length-prefixed framing on a `SOCK_STREAM` Unix socket:
- Client connects, then per inference: send `count` (4 bytes) + planes payload.
- Server replies: `count` (4 bytes) + policy payload + value payload.
- Read loops must handle partial reads (`recv` may return fewer bytes).

Simple, robust, trivially cross-language. Copy cost is negligible at batch
granularity. Use this until measured to be a bottleneck.

### Transport B — shared memory (v2, optional optimization)

One shared segment, a ring of `S` slots (S=2–4 for pipelining), each sized for
`MAX_BATCH` (e.g. 512):

```
┌─ control block ───────────────────────────────────────────┐
│ magic u32 │ max_batch u32 │ num_slots u32 │ shutdown u32   │
├─ slot headers [S] ────────────────────────────────────────┤
│ state u32 {0 FREE,1 REQ,2 RESP} │ count u32 │ seq u64      │
├─ request regions  [S] :  planes[max_batch*486] f32         │
├─ response regions [S] :  policy[max_batch*209] + value[..] │
└────────────────────────────────────────────────────────────┘
```

Two POSIX named semaphores: `sem_req` (Go posts, Python waits),
`sem_resp` (Python posts, Go collector waits). Slot state machine
(single writer per transition):

```
 FREE ─(Go: write planes,count; state=REQ; post sem_req)─► REQ
 REQ  ─(Py: wait sem_req; forward; write results; state=RESP; post sem_resp)─► RESP
 RESP ─(Go: wait sem_resp; read; scatter to games; state=FREE)─► FREE
```

Semaphore post/wait provide the release/acquire barriers, so planes written
before `post(sem_req)` are visible after `wait(sem_req)`. The free-slot pool
lives inside Go (a buffered channel of indices). Response routing (which games’
leaves occupy which rows of a slot) stays in Go memory; shared memory carries
only planes/results.

## 3. Weight sync

Learner writes `checkpoints/current.pt` atomically (`write tmp → os.replace`).
Inference checks the file mtime between batches; on change, `load_state_dict`
into a spare net and swap. Slightly stale weights during self-play are fine
(standard in async AlphaZero).

## 4. Invariants / risks

- **Engine parity** is the top risk: Go `encode_state`, action indexing, and
  wall-legality must match Python exactly. Enforced by golden vectors +
  differential tests.
- The board encoding and action layout are **frozen** once a large dataset is
  generated — changing them invalidates stored shards and the warm-start path.
