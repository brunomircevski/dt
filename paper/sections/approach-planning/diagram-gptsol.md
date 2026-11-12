# Diagram proposal: Hybrid level and hand-off

## (a) Decision and rationale

Include one diagram. The hybrid schedule is the paper's main architectural
contribution and is harder to recover from prose than the serial or multicore
deltas. A compact figure can show three facts simultaneously: assignment is
per child node by size, the hand-off is asynchronous, and the CPU and GPU
contribute to the same tree. It should not become a CUDA kernel pipeline.

## (b) What the figure must show

- A GPU box containing one current level with several large nodes.
- A split by node size after the level: children at or above
  `gpu-min-rows` form the next GPU level; smaller children follow the hand-off
  path.
- The hand-off path in this order:
  **device ring → pinned staging buffers / copy streams → pageable host arena
  → CPU thread-pool subtree tasks**.
- A visual overlap: while CPU tasks grow handed-off subtrees, the GPU processes
  the next level. A light vertical time direction or parallel swim lanes is
  sufficient.
- Both paths ending in one shared/concurrent node store, followed by preorder
  assembly.
- A small annotation near staging: “buffers and copy threads on GPU NUMA
  node”. A separate annotation near the arena: “pre-touched; no explicit NUMA
  binding”.
- A note that \ImplGpu{} is the same GPU lane with the threshold set to one
  row, hence no nonempty child takes the CPU path.

## (c) What to omit

- Kernel names, tile dimensions, warp operations, radix-sort internals, and
  the single- versus double-precision sweep.
- Exact ring and staging-buffer capacities; they are implementation constants,
  not necessary to understand the schedule.
- Performance numbers or claims that either device is always faster.
- Forest-level parallelism, multiple GPUs, distributed workers, or any
  socket-local CPU scheduler.
- An arrow suggesting that the pageable host arena is allocated on the GPU's
  NUMA node. Only the pinned staging buffers and dedicated copy threads have
  that explicit placement.
- A full decision tree. Two levels and two representative CPU subtrees are
  enough.

## (d) Caption draft

**Size-aware growth in \ImplHybrid{}.** The GPU processes the large nodes of a
level while small children are gathered, copied through pinned staging buffers
into a host arena, and submitted to the CPU pool. Host callbacks overlap CPU
subtree growth with the next GPU level; both paths contribute to one tree
store.

## (e) Concrete sketch

Draw as two horizontal swim lanes with time increasing from left to right:

```text
                              child size >= m
GPU lane    [level d: large nodes] ───────────────► [level d+1: large nodes] ──┐
                 │ split children                         │                    │
                 │ child size < m                         │ continues          │
                 ▼                                        │ concurrently       │
handoff      [device ring] → [pinned staging] → [pageable host arena]          │
                                  │                     │                      │
                          GPU NUMA node          pre-touched, not bound         │
                                                        ▼                      │
CPU lane                                           [CPU subtree tasks] ────────┤
                                                                              ▼
                                                               [shared node store]
                                                                      │
                                                                      ▼
                                                             [preorder assembly]
```

Suggested visual encoding:

- GPU lane: blue boxes.
- CPU lane: neutral grey or green boxes.
- Communication path: orange arrows.
- Solid arrows: data or node flow.
- A thin dashed bracket spanning “next GPU level” and “CPU subtree tasks”:
  “concurrent”.
- Put the threshold \(m=\texttt{gpu-min-rows}\) beside the branch, not in a
  legend.
- Add a small footnote inside the figure:
  “\ImplGpu{}: \(m=1\), so no nonempty subtree is handed off.”

The final asset should be exported as:
`paper/figs/approach-hybrid.pdf`.
