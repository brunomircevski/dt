# Section 3 plan — Round 2 peer review and consensus vote (Muse Wonder)

Plan only. No section prose; `approach.tex` untouched.

Scores are 1 (poor) – 5 (excellent), judged for the *writing task as briefed*:
~1,200–1,800 words, CCPE/LNCS tone, fact-accurate from `src/` + `docs/`,
NUMA only where implemented, shared semantics not re-taught.

## 1. Scores

| Plan | Structure | Fact caution (esp. NUMA) | Completeness vs author request | Length realism | Overall impression |
|---|---:|---:|---:|---:|---|
| **Opus** | 4 | 4 | 5 | 3 | Richest GPU/Hybrid treatment and a useful overview table, but ~1,920 words *plus* a table *plus* an algorithm box is not a 2–3 page plan without the cuts it only lists as fallback. |
| **GPT-Sol** | 5 | 4 | 4 | 5 | Cleanest allocation map (prelude vs deltas) and the most disciplined budget; slightly thinner on GPU setup/overlap and on why Hybrid is *not* a static partition. |
| **Muse (self)** | 4 | 3 | 5 | 3 | Most exhaustive traceability and omission list, but overloaded: it uses "proven bound" language the docs themselves qualify (Risks A/B), and its 3.1 bullet overstates external agreement as byte-identical against scikit-learn/Quinlan, where `docs/ALGORITHMS.md` documents tie/rounding exceptions. Its 1,600-word + figure + optional-table arithmetic has no slack. |
| **Grok** | 4 | 5 | 5 | 4 | Best fact discipline of the four: catches that `--serial` presort is single-threaded (null pool), that the `options.h` "--gpu-only --threads 1 ⇒ single-threaded copies" comment is stale vs `kMinCopyThreads = 8`, that root class counts on the CUDA path are a *device* histogram, and that the Hybrid arena is first-touched by the pool and *not* bound to the GPU node. Only structural wart: putting the NUMA heading in the Hybrid subsection when the staging placement serves both GPU modes. |

### Notes behind the scores (only points that change the consensus)

- **Opus — fact caution 4, not 5:** correct headline ("no explicit NUMA policy;
  reliance on first-touch; do not claim NUMA-aware scheduling"), but
  "pages end up spread over the sockets roughly as the work is" is an
  inference from comments, stated a shade too confidently; likewise
  "the experiments do not prune" generalises the depth-≤10 protocol in the
  captions it cites. Both are one-clause fixes, not structural flaws.
- **GPT-Sol — fact caution 4, not 5:** exemplary on external-tree caveats
  ("documented exact ties", FP32-vs-FP64 edge cases), the boundary-point
  Risk A, and "avoid claiming a formal proof". Loses a point for small
  over-generalisations a writer could copy verbatim — e.g. a one-bit-per-row
  bitmap presented as *the* Serial Optimized marking scheme (the CPU builder
  uses the byte map for in-place nodes and the bitmap for the out-of-place
  path), and "same double-precision `cutGain`" without the estimate/filter
  qualifier until later.
- **Grok — structure 4, not 5:** subsection 2.6's title advertises "the only
  NUMA placement in the source", yet its own text (correctly) says the
  placement applies to uploads/downloads of *both* `\ImplGpu` and
  `\ImplHybrid`. A reader skimming headings would misfile it as Hybrid-only.
- **All four agree, and I accept:** six subsections (shared prelude + five
  configurations); `\ImplGpu`/`\ImplHybrid` are modes of one CUDA grower and
  this must be said once; no per-optimisation speed-up attribution (no
  ablation exists); no kernel-name catalogue; pruning semantics stay out,
  beyond the grower→pruner interface; CPU NUMA = affinity mask + huge pages
  only, GPU staging = the one explicit placement.

## 2. Consensus structure I accept

Lead-in (unnumbered, ~70 words, no table) + six subsections.
**One float only:** a single schematic/algorithm figure serving §3.5–3.6
(level step + hand-off). If the figure is dropped, restore ~120 words to
3.5/3.6; if an overview table is wanted instead, it *replaces* the figure
and the lead-in shrinks — not both floats.

| # | Title | Budget | Key content (bullets, not prose) |
|---|---|---:|---|
| — | Lead-in | ~70 | Five configurations, one exact search, one tree; they differ in how sorted order is obtained/maintained, how the best split is found, and where nodes are processed. Flags named once here. |
| 3.1 | Shared Search and Column Layout | ~250 | Column-major `float` data, integer class ids; shared `cutGain`/`SplitRules`, `1e-12` tie → earlier cut → lower feature; shared `c·log₂c` table; optimised builders' invariant: node = same index range in every presorted 8-byte-entry column, stable partition ⇒ sort once (SLIQ/SPRINT cross-ref); flat preorder tree from chunked store + renumbering; shared post-processing in one sentence, growers record distinct values / leaf-of-row; internal identical-tree claim + golden checks in ≤2 sentences, external caveats *not* restated here. |
| 3.2 | `\ImplSerialBase` (Serial Simple) | ~140 | Textbook reference: per node/feature copy row ids, `std::sort`, exact gain at every admissible cut, two fresh row vectors, preorder recursion; `O(F·n log n)` per node + gathers + per-node allocation as the cost being designed away; same rules/pruners as all others; fills neither distinct values nor leaf-of-row (trainer recomputes). No recursion-depth caveat. |
| 3.3 | `\ImplSerialOpt` (Serial Optimized) | ~350 | Three moves, each with its tree-preservation clause: (1) stable LSD radix presort once (single-threaded here — null pool), split by position (`leftCount`), stable partition of all columns, out-of-place ≥ `2^15` rows into 1 spare slot / in-place below, terminal children skip partition, constant features dropped subtree-wide; (2) filtered sweep — incremental estimate, exact `cutGain` only for candidates settled in order, two-class specialisation, 16-cut convexity block bound (≥64 rows), step bound, equal-value runs absorbed, children's counts from winning sweep, fused child sweeps (binary, ≥4,096); boundary-point skip named only as a restricted candidate-elimination test (see Disagreement 4); (3) memory — `(F+1)·8N` slots allocated once, count tables `16N` for `N ≥ 65,536`, ≥2 MiB allocations on transparent huge pages, explicit stack. Ends: "this builder with the pool switched off" ⇒ 3.4 is a pure delta. |
| 3.4 | `\ImplParallel` (Multicore CPU) | ~260 | Same algorithm, pool added. Data parallelism ≥32,768 rows (feature sweeps, one column per thread in partition; spare slots `min(F−1, threads)`; blocked partition ≥ `2^20` rows when slots are scarce); task parallelism ≥4,096 rows (left child as task, right inline, one final wait); deadlock-free nested `parallelFor`; parallel presort (incl. splitting one column across threads when `F < threads`) and parallel preorder assembly, one sentence each. **NUMA scoping (2–3 sentences):** no pinning, no socket detection, no per-socket data; thread count respects the process affinity mask (`taskset`/`numactl` honoured); huge pages only. Thresholds stated as tuned constants, not laws. |
| 3.5 | `\ImplGpu` (GPU-Based, `--cuda --gpu-only`) | ~320 | Pure form of the CUDA grower: breadth-first levels vs CPU depth-first; segments/tiles of 2,048 entries, one block per tile; five-stage level pipeline (constant-segment detection → histograms/prefix, root-only below level 0 → sweep → per-(node,feature) best + **one host round trip per level** for shared `SplitRules::choose` → smaller-side bitmap marking → single-pass stable partition that also emits next level's histograms); warp ballot/shuffle sweep; precision: FP64 one-pass on fast-FP64 parts (the evaluated mode — say so), else conservative FP32 filter + exact rescoring, tile convexity bound for binary; device `F+1` slots, CUB presort overlapped with staged upload (4×32 MiB pinned buffers, copy stream, context prepared during load); **NUMA here, not in 3.6:** staging buffers allocated on and copy threads pinned to the GPU's sysfs NUMA node, both GPU modes, ≥8 copy threads even at `--threads 1`; CPU still stages copies and runs C4.5 pruning — no CPU subtrees, no arena; device-capacity check `(F+1)·8N` with `--parallel` fallback in one sentence. |
| 3.6 | `\ImplHybrid` (Hybrid, `--cuda`) | ~240 | Delta from 3.5 only: children below `m` leave the GPU; auto `m` = `N/8192` clamped [1024, 2048] (CART) / `N/4096` clamped [1024, 4096] (C4.5), user-overridable, size is the *only* criterion; rationale: level cost is weakly size-dependent on GPU, small subtrees are cheap CPU tasks. Hand-off: gather still-splittable columns into 512 MiB device ring (4 parts) → two 128 MiB pinned buffers on two copy streams → pageable host arena (`8FN`, each row handed over ≤ once, background pre-touch, wait only for the part being filled) → `cudaLaunchHostFunc` starts CPU tasks, GPU proceeds; shared node store + preorder renumbering keeps one identical tree; `\ImplGpu` = degenerate threshold, a training set below `m` is CPU-grown (one clause); arena memory cost acknowledged in one sentence. No "static root partition / racing implementations" misreading — assignment is per child, per level. |
| | **Total** | **~1,630** | ~2.5 pages prose; ≈2.8–3.0 pages with the single float. Hard ceiling 1,800: first cuts are 3.3 move-(2) detail and 3.4 presort/assembly sentences. |

## 3. Remaining disagreements (max 5) and my preferred resolution

1. **Where the 8-byte entry is introduced — 3.1 (GPT-Sol, Muse, Grok) vs
   3.3 (Opus Q1).** *Preferred:* **3.1**, as consensus above. The entry +
   node-range invariant is the vocabulary 3.3–3.6 all spend; introducing it
   in 3.3 forces 3.1 to describe the layout twice (once vaguely, once
   concretely). Opus's "progression story" concern is met by keeping the
   *radix presort mechanics* in 3.3 — the entry is representation, the sort
   is the optimisation.
2. **Floats: overview table (Opus) vs hybrid algorithm/schematic (Muse,
   Opus alt.) vs none committed (GPT-Sol, Grok).** *Preferred:* **one
   visual — the level-step/hand-off schematic (or CUDA.md Algorithm 1 cut
   to ~12 lines) — and no overview table.** The table duplicates 3.1 prose
   and, per Opus's own arithmetic, is what pushes the section past 3 pages;
   the Hybrid schedule is the contribution and the only mechanism prose
   alone makes hard to follow. Table only if the author insists, *instead*
   of the figure, funded by trimming the lead-in.
3. **Where GPU-staging NUMA lives — GPU subsection (Opus, GPT-Sol, Muse)
   vs Hybrid subsection (Grok's heading, against its own text).**
   *Preferred:* **3.5, explicitly "both GPU modes", cross-referenced in
   one clause in 3.6.** The source (`gpu_host.cuh: Staging`) serves uploads
   and downloads regardless of hand-offs; filing it under Hybrid teaches
   exactly the wrong mental model. Grok's factual content is adopted
   wholesale; only its placement is not.
4. **Boundary-point shortcut and "proven/provably" language — feature it
   with Fayyad–Irani framing (Opus, Muse) vs near-silence (GPT-Sol,
   Grok).** *Preferred:* **GPT-Sol/Grok position.** Name it at most as a
   restricted same-class candidate-elimination test inside 3.3's sweep
   list, without a Fayyad–Irani theorem claim, and use "conservative
   bound / candidates that cannot beat the incumbent under the tie rule"
   everywhere instead of "proven/provably" — for the GPU FP32 filter as
   well. `docs/SIMPLE.md` Risks A/B (tie-rule interaction; bounds argued
   in comments, no adversarial test) make stronger wording a hostaged
   claim; the identical-tree *evidence* (golden tests) carries the point
   safely in 3.1.
5. **Scope of the shared prelude on pruning — pipeline sentence only
   (all four, differing in warmth) vs a short shared paragraph (Muse Q6
   fallback), incl. CART's α early-stop.** *Preferred:* **one sentence in
   3.1 + the grower→pruner interface line; no α early-stop, no C4.5
   pruning engineering.** Results captions frame the headline protocol as
   unpruned, pruning internals are algorithm semantics already placed in
   §2.1, and every plan's omission list independently converges here.
   Revisit only if the author confirms pruned timings are a headline
   result — then the fix is +80 words in 3.1 taken from 3.3, as Muse
   proposed, not a new subsection.

*Not counted as disagreements (already unanimous, recorded so they are not
re-litigated in writing):* six-subsection shape; one-CUDA-grower disclosure;
no ablation-free speed-up attribution; serial presort is single-threaded;
CPU has no NUMA policy to describe; external-reference agreement carries
documented tie/precision caveats and §3 claims identity *among our five
configurations*.
