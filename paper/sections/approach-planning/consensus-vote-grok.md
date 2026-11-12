# Consensus vote — Section 3 plan (Grok)

Vote only. No section prose. `approach.tex` was not edited.

Scores are 1–5. “Peer” plans are Opus, GPT-sol, and Muse. Grok is this agent’s own plan, scored on the same scale.

## 1. Scores

| Plan | Structure | Fact caution (NUMA) | Completeness vs the brief | Length realism |
|---|---:|---:|---:|---:|
| Opus | 5 | 4 | 5 | 3 |
| GPT-sol | 4 | 5 | 4 | 5 |
| Muse | 5 | 4 | 5 | 4 |
| Grok (own) | 4 | 5 | 4 | 5 |

**Opus.** Clearest “each version changes the previous one” narrative, and the NUMA split (no CPU policy; staging buffers on the GPU node) is right. The recommended draft is about 1,920 words plus a table and an algorithm box, which the plan itself puts past three pages. Content bullets still say shortcuts “provably” cannot win and that the Fayyad–Irani rule is applied “where it is valid under the tie rule”; `docs/SIMPLE.md` Risks A and B do not support that wording. “Pages end up spread over the sockets roughly as the work is” is stronger than first-touch.

**GPT-sol.** Best restraint: no formal proof, no CPU NUMA, external reference checks kept caveated, boundary shortcut optional. Structure is the same six subsections but thinner on what actually changes (partition by position, spare-slot counts, warp sweep). Word range 1,450–1,650 matches the author’s 1,200–1,800.

**Muse.** Best map of what is said once versus per backend, and the NUMA scoping warning is the one a writer cannot miss. Subsection bullets still say “proven” bounds, “bit-identical” against scikit-learn and Quinlan, and “half the PCIe rate”. Default 1,600 words plus a figure and an optional table has no slack. Page math at ~800–850 words per LNCS page also disagrees with the author’s own 1,200–1,800 ≈ 2–3 pages; plan to the author’s conversion.

**Grok (own).** Strongest list of traps (serial presort is single-threaded, `--threads 1` does not limit staging copies to one thread, arena is not on the GPU node, do not claim bit-identical intermediate estimates). Parking the only NUMA mechanism in the Hybrid subsection is the wrong place: uploads use it in both GPU modes. The 400-word Serial Optimized budget invites the sweep tour the plan itself says to cut. No comparison figure or table, which the other plans use to show how the five differ.

All four already agree on the skeleton: one shared subsection, then the five macros, no CART lecture, no benchmark tables, no invented CPU NUMA scheduler.

## 2. Consensus structure I accept

Prose **about 1,530 words**. No overview table and no algorithm box in the first draft. At most one schematic, and only if the body is cut to about 1,400 words. The author’s conversion (1,200–1,800 words ≈ 2–3 LNCS pages) is the length target, not an 850-word page.

| Subsection | Words | Role |
|---|---:|---|
| 3.1 Common design | 220 | Once: scoring, input layout, entry invariant, tree, pruning pointer |
| 3.2 Serial Simple | 140 | Textbook baseline |
| 3.3 Serial Optimized | 360 | Presort, filtered sweep, slots |
| 3.4 Multicore CPU | 240 | Same algorithm; two schedules; no NUMA policy |
| 3.5 GPU-Based | 320 | Levels, device memory, exact sweep, staged copies, GPU-node pinning |
| 3.6 Hybrid | 250 | Size hand-off and overlap only |
| **Total** | **1,530** | |

Titles can stay this short. Descriptive subtitles are optional and must not add a seventh topic.

### 3.1 Common design (220)

- Two or three sentences: five named implementations, two modes of one CUDA grower (`--cuda --gpu-only` vs `--cuda`), same exact numeric binary search, same tree among our implementations. Not five independent programs.
- Column-major `float` values, integer class ids.
- Except Serial Simple: 8-byte entries `(value, packed row and class)`; a node is one index range in every column; a stable partition yields two sorted ranges, so each feature is sorted once. Cite SLIQ/SPRINT in related work; do not retell it. Mechanics of the sort and the partition wait until 3.3.
- One scoring function, compiled for CPU and GPU. Ties within \(10^{-12}\): earlier cut, then lower feature index. Small counts use one \(c\log_2 c\) table. The selected cut is that exact gain. Do not say every intermediate estimate is bit-identical, and do not say “byte-identical to Quinlan” here.
- Flat preorder tree. Chunked concurrent store, then renumber, so creation order does not change the tree. One sentence is enough; 3.4 only points back.
- Pruning is shared and after growth. One sentence. Optimized growers can record leaf ids and distinct values. No raising, no MDL derivation.

### 3.2 Serial Simple (140)

- One thread, depth-first: per node and feature, copy row ids, comparison-sort, exact gain at every admissible cut, two new id vectors, recurse.
- Same rules and pruners. No presort, no packed entries, no estimate filter, no constant-feature drop.
- Cost in complexity only: repeated \(O(Fn\log n)\) and gathers by row id. It is the baseline the later subsections change.
- Distinct values and row routing for C4.5 are computed outside this grower. Omit the recursion-depth caveat.

### 3.3 Serial Optimized (360)

- Same builder as Multicore with a null pool, including out-of-place partition. Presort on this path is single-threaded. Do not import the parallel-presort paragraphs from `docs/CPU.md`.
- One stable LSD radix presort per feature, then a linear sweep and a stable partition. Level cost \(O(Fn)\) instead of sorting again. Order inside a run of equal values does not affect the cut.
- Partition is by position on the winning column (first `leftCount` entries). Other columns follow a goes-left mark. Nodes of at least \(2^{15}\) rows move out of place through one spare slot; smaller nodes use a reused buffer. Both-leaf children skip the move. Constant features (C4.5: within \(10^{-5}\)) leave the subtree.
- Sweep: incremental estimate, exact `cutGain` only for cuts that can change the winner under the tie rule, in order. Two-class specialization and a convex block bound on large binary nodes. Do not list block size 16, the step bound, lane counts, or the 256-cut refresh unless the paragraph is still short. Do not name Fayyad–Irani. Do not say “proven” or “provably”.
- Memory: \(F+1\) slots, \((F+1)\cdot 8N\) bytes, allocated once; transparent huge pages for large blocks (TLB, not placement). No per-item speed-ups.

### 3.4 Multicore CPU (240)

- Deltas only. Data parallel at \(\ge 32{,}768\) rows (features, then columns). Subtree tasks at \(\ge 4{,}096\) rows (left child is a task, right continues inline, one wait at the end). Nested `parallelFor` cannot deadlock the pool.
- Spare slots \(\min(F-1,\text{threads})\). Nodes of at least \(2^{20}\) rows with fewer free slots than threads are partitioned in blocks. Parallel presort belongs here.
- **NUMA, negative claim, two sentences.** The pool does not pin workers, bind memory, or replicate columns per socket. Thread count follows the process affinity mask, so an external `taskset` or `numactl` is honoured. Large arrays are first-touched by the threads that write them. Huge pages are not a NUMA policy. Socket scaling is an experimental result, not a feature of this section.

### 3.5 GPU-Based (320)

- Depth-first on the CPU, one level at a time on the GPU. Tiles of 2,048 entries, one block per tile. Same entries and \(F+1\) slots on the device. No hand-off arena in this mode.
- One host round trip per level: best cut per feature comes back; the host runs the same feature choice as the CPU. Leaves close without a partition. Partition is stable and single-pass; the smaller side is marked; the winning column is split by position; the write counts the next level’s histograms.
- Fast double precision: score every cut once with `cutGain`. Otherwise a single-precision filter, then double precision only for survivors. The filter is only allowed to drop cuts that cannot change the winner. Evaluated data-center GPUs take the one-pass path under `auto`. Two-class tiles can be skipped by a corner bound. No kernel names.
- Upload through a few page-locked staging buffers while the source stays pageable. Device radix presort overlaps the upload. Root class counts are a device histogram, not a host pass.
- **NUMA, positive claim, here.** If the GPU’s PCI NUMA node is known and the process may also run elsewhere, those staging buffers are allocated on that node and the copy threads are pinned to its CPUs. This serves both GPU-Based and Hybrid. It does not place the training matrix or the hand-off arena. With `--threads 1`, build-side host work is one thread except those copy threads, which still use at least eight when the node is known (`gpu_host.cuh`, not the stale `options.h` comment).

### 3.6 Hybrid (250)

- Same level loop. A child below \(m\) rows is gathered (only columns that can still split) and copied to the host. The GPU continues. A host callback starts CPU tasks, which use the Multicore builder. One wait at the end.
- \(m\) is `--gpu-min-rows` or the automatic clamp: CART \(N/8192\) into \([1024,2048]\), C4.5 \(N/4096\) into \([1024,4096]\). Reason: a level of many tiny nodes is a poor GPU fit; say the defaults are empirical. No sensitivity table.
- Communication: 512 MB device ring, two pinned download buffers, pageable arena large enough that each row is handed over at most once, pre-touched so the copy does not fault. The GPU waits only when it is a full ring ahead of the bus. The arena is first-touched by the CPU pool and is **not** bound to the GPU node.
- GPU-Based is this grower with hand-off disabled (threshold 1, no arena). Do not describe removed GPU pruning.

## 3. Disagreements still to resolve

Five, with the resolution I would write to.

1. **Where the 8-byte entry is defined.** Opus would introduce it in Serial Optimized so the layout change is that subsection’s optimization. The other three define it in the shared subsection. **Prefer the shared subsection, with an explicit “not Serial Simple”**, and keep radix sort and partition mechanics in Serial Optimized. The GPU and multicore subsections then point back instead of redefining the entry.

2. **Where GPU NUMA is stated.** Grok puts pinning in Hybrid. Opus, GPT-sol, and Muse put it in GPU-Based, which is the section that introduces the upload. **Prefer GPU-Based**, one cross-reference from Hybrid. Hybrid’s only NUMA sentence is that the arena is not placed on the GPU node.

3. **Table, algorithm box, or figure.** Opus wants a comparison table and an algorithm (~3.2 pages). Muse wants a two-panel figure and trims prose only if both a figure and a table are kept. Grok allows one schematic and forbids a second float. GPT-sol needs neither. **Prefer no table and no algorithm box.** One schematic of column ranges plus the hand-off is allowed only if prose drops to about 1,400 words. Otherwise the six subsections are the comparison.

4. **How certain the sweep filters are.** Opus and Muse’s content bullets say “provably” / “proven” / “the bound guarantees”. GPT-sol and Grok refuse a formal proof and treat Fayyad–Irani as unsafe to feature. **Prefer: “a bound rejects cuts that cannot change the selected split; the accepted cut is an exact `cutGain`.”** Omit the boundary-point shortcut. Do not cite Fayyad–Irani as a theorem this code fully applies. Related work already states the classical result.

5. **“The same tree” against scikit-learn and Quinlan, inside Section 3.** Opus and Muse would repeat byte-identical external checks in the shared subsection. GPT-sol and Grok limit Section 3 to agreement among our implementations and leave the documented tie and single-precision exceptions in the introduction. **Prefer the narrower claim.** One clause may point at the introduction. Do not write “byte-identical to Quinlan” in this section.

Not a sixth fight, but every draft should also keep these, which are not in dispute once stated: serial presort is single-threaded; root class counts on the CUDA path are a device histogram; no per-optimization speed-ups; no CPU socket scheduler.
