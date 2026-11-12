# Section 3 Our Approach — Agreed Plan (for author review)

Agents: Opus, GPT Sol, Muse Wonder, Grok (fresh; code+docs grounded).
Status: **converged after Round 2**. No prose drafted; `approach.tex` unchanged.
Target length: **~1,500–1,650 words (~2.5–3 LNCS pages)**. Optional: one Hybrid hand-off figure.

Paper macros: `\ImplSerialBase`, `\ImplSerialOpt`, `\ImplParallel`, `\ImplGpu`, `\ImplHybrid`.

---

## Agreed structure

| § | Title | Words | Role |
|---|---|---:|---|
| lead-in | (inside 3.1 or 2–3 sentences before it) | ~70 | Five configurations; four code paths (GPU-only and Hybrid = two modes of one CUDA grower) |
| 3.1 | Common Design: Exact Splits and Column Layout | ~220–250 | Shared scoring, layout, tree store, verification/pruning one-liners |
| 3.2 | Serial Simple | ~140 | Textbook baseline / correctness reference |
| 3.3 | Serial Optimized | ~350–380 | Presort + filtered sweep + memory discipline |
| 3.4 | Multicore CPU | ~240–260 | Feature- and task-parallel deltas; NUMA scope statement |
| 3.5 | GPU-Based | ~320–340 | Level-wise GPU pipeline; GPU staging NUMA |
| 3.6 | Hybrid | ~240–250 | Size-based device assignment + async hand-off |

No overview table. No algorithm/pseudocode box. At most one figure (Hybrid level-step / hand-off).

---

## Per-subsection content

### Lead-in
- Five implementations grow the **same** exact CART/C4.5 tree; they differ in (i) how sorted order is maintained, (ii) how candidate cuts are filtered/scored, (iii) where and in what order nodes grow.
- `\ImplGpu` and `\ImplHybrid` are two configurations of one CUDA grower (`--gpu-only` vs default `--cuda` with CPU hand-offs).

### 3.1 Common Design
- Column-major float32 features; labels remapped to dense integers.
- Shared scoring (`cutGain` / impurity) compiled for CPU and GPU; fixed deterministic tie rule (near-equal gains → earlier cut, then lower feature index).
- Optimizations may only discard cuts that cannot win under the criterion (no “approx tree”).
- Presorted 8-byte entries (value + class) and a node as a contiguous index range — used by all optimized backends; Serial Simple does **not** use this layout.
- Flat preorder tree store / renumbering so output is deterministic under parallelism.
- One sentence: identical trees across our five implementations (checked); reference checks vs scikit-learn / Quinlan already in Intro — do not restate caveats here.
- One sentence: grow then prune; Experiments mainly use unpruned / depth-limited settings as stated in design section.

### 3.2 Serial Simple (`--simple`)
- Textbook: at each node, for each feature, sort row ids by value and evaluate every admissible cut.
- Fresh allocations / recursive growth; memory-latency bound (indirect value reads).
- Role: executable specification and oracle for the others.

### 3.3 Serial Optimized (`--serial`)
Three moves, each with a “why tree unchanged” clause; **no per-optimisation speed-up claims** (no ablation):
1. **Presort once** (radix) into 8-byte entries; grow by **stable partition by position**, not re-sort. Serial presort is **single-threaded**.
2. **Filtered exact sweep**: cheap estimate / convexity / constant-feature dropping; only surviving candidates scored exactly with shared math. Prefer “conservative bounds” — do **not** say “proven”; do **not** name Fayyad–Irani unless citing carefully.
3. **Memory discipline**: slots allocated once; huge pages; explicit stack (no deep recursion).
- Bridge: same builder as Multicore with the thread pool off / one worker.

### 3.4 Multicore CPU (`--parallel`)
- Same layout and sweep as Serial Optimized.
- **Data parallelism**: features of a node in parallel when node ≥ **32,768** rows.
- **Task parallelism**: subtrees as pool tasks when ≥ **4,096** rows.
- Spare column slots so features can partition concurrently; parallel presort; concurrent node store then preorder renumber for determinism.
- **NUMA / sockets (scope, not a feature):** thread count follows process affinity (`taskset`/`numactl`); memory follows Linux first-touch / huge pages. **No** libnuma pinning, interleaving, or per-socket pools in main `src/`.

### 3.5 GPU-Based (`--cuda --gpu-only`)
- Level-synchronous growth: all active nodes of a level processed together, in tiles.
- Device-side pipeline (high level): prepare → sweep → reduce best cut → partition → emit children; **one host round-trip per level**.
- Exact scoring with shared double-precision path on evaluated A100/H100 hardware; float pre-filter is optional / not the evaluated mode — one cautious sentence max.
- Uploads via pinned staging buffers; **only explicit NUMA placement in the code**: staging/copy buffers bound to the **GPU’s NUMA node** (serves both GPU modes — place the NUMA fact **here**, not only under Hybrid).
- GPU-only: hand-off threshold effectively **1 row** (no useful CPU subtree hand-off). Do not claim “threshold 0”.
- Avoid naming every kernel; no timing numbers.

### 3.6 Hybrid (`--cuda`, default)
- Assign nodes by size: large nodes stay on GPU (level-wise); children below `--gpu-min-rows` go to the CPU pool. Defaults scale with dataset size / were chosen empirically.
- Async hand-off: device ring → pinned host buffers → host arena; CUDA host callback starts CPU tasks; GPU continues without waiting for CPU finishes.
- Host arena is **pre-touched** by CPU threads; it is **not** explicitly bound to the GPU’s NUMA node (Grok-verified).
- Both devices write into one shared node store; final tree still identical.
- Clarify what Hybrid is **not**: not forest-level device split; not CPU-control/GPU-offload only.

---

## Fact discipline (all agents agree)

| Do | Don't |
|---|---|
| Say four code backends / five experimental configurations | Invent a fifth independent grower |
| Limit explicit NUMA to GPU staging on GPU’s node | Claim NUMA-aware CPU scheduling |
| “Conservative” sweep filters | “Proven” / “provably optimal skip” without proof |
| Identical trees across our implementations | Over-claim byte-identity vs scikit-learn/Quinlan beyond Intro |
| Qualitative “why” for optimisations | Attribute ×-speedups to individual opts (no ablation) |
| Place staging NUMA in §3.5 | Put it only under Hybrid |

---

## Omit from Section 3

- Benchmark / ablation numbers (belong in Results)
- Kernel-by-kernel CUDA walkthrough and NVTX/profiling tooling
- Failed experiments / tuning archaeology (PERFORMANCE.md history)
- Full CART/C4.5 textbook re-teach (cite §2.1)
- Detailed C4.5 pruning / MDL internals
- Stale README/`options.h` comments that contradict code (e.g. `--gpu-only` copy-thread note)

---

## Open author questions (non-blocking for plan approval)

1. Include the optional Hybrid hand-off figure now, or prose-only first?
2. Confirm Experiments’ CPU affinity / GPU-node binding narrative matches §3.4–3.5 wording (chart captions vs text).
3. Preferred page budget: lean ~1,500 words or allow ~1,650 with figure?

---

## Peer rating snapshot

| Agent | Favoured peers | Emphasis |
|---|---|---|
| Opus | GPT-Sol 4.5; Muse/Grok 4.25 | Corrected own GPU-only threshold=0 error → threshold 1 |
| GPT Sol | All strong; own length discipline | 1,500–1,600 words; GPU staging NUMA scoped |
| Muse | Grok best fact caution | Hybrid figure only; no “proven” |
| Grok | GPT-Sol/Muse length; Opus structure | Serial presort single-thread; arena not GPU-node-bound |

Consensus merges: GPT-Sol/Muse length + shared-vs-delta map; Grok fact catches; Opus GPU/Hybrid mechanism detail (trimmed).
