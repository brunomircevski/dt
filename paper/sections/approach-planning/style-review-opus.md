# Style review of Section 3 (Opus)

Scope: `approach.tex` (primary), checked against `introduction.tex` and
`related-work.tex` for style and terminology. No paper files were edited.

Summary of findings:

- Intro and Related Work already use "samples" and "attributes" throughout.
  Section 3 is the only section (outside `approach-planning/`) that uses
  "rows" / "features" (14 occurrences, all in `approach.tex`).
- Section 3 says "five configurations"; the Introduction says "five
  implementations". Recommend "implementations" everywhere.
- One CLI flag remains in Section 3 (`--cuda --gpu-only`), plus a flag list in
  a `\plan{}` note in `experiment-design.tex` (see table at the end).
- Section 3 uses both "page-locked" and "pinned" for the same buffers.
- Related Work uses British spelling ("parallelised", "organise") while the
  Introduction uses American ("organized").

---

### S-Opus-1
- Location: `approach.tex`, opening paragraph, "Five configurations grow the same exact CART or C4.5 tree. ... without and with hand-offs of small subtrees to the CPU."
- Issue: terminology (configurations/rows/feature) + style (one dense 14-line paragraph)
- Proposed rewrite:
```latex
Our five implementations grow the same exact CART or C4.5 tree. They differ
in how the samples of a node are kept sorted, which cuts are scored exactly,
and where and in which order nodes are grown. Attribute values are stored
column-major as 32-bit floats, and class labels are mapped to dense integers.

All implementations use the same CART and C4.5 rules and the same
double-precision gain calculation, compiled for both CPU and GPU. Gains within
$10^{-12}$ count as ties; the earlier cut wins, then the attribute with the
lower index. Optimizations may discard only cuts that cannot change this
decision. Concurrent builders place nodes in a shared store and renumber them
into flat preorder arrays at the end, so the order of creation cannot alter
the output. Our tests require identical trees from all five implementations.
When pruning is enabled, shared CPU code runs it after growth. \ImplGpu{} and
\ImplHybrid{} are two modes of one CUDA grower, without and with hand-off of
small subtrees to the CPU.
```
- Priority: high
- Note: if the authors deliberately chose "configurations" because \ImplGpu{}
  and \ImplHybrid{} share one grower, the last sentence already explains this,
  so "implementations" stays accurate and matches the Introduction.

### S-Opus-2
- Location: `approach.tex`, §\ImplGpu, "\ImplGpu{} (\texttt{--cuda --gpu-only}) sets the hand-off threshold to one row, ... runs shared post-processing."
- Issue: flag (author requirement 2) + terminology + shorten (the last sentence repeats that the host chooses splits)
- Proposed rewrite:
```latex
\ImplGpu{} sets the hand-off threshold to one sample, so every nonempty
subtree remains on the GPU. Unlike the depth-first CPU builders, it processes
all active nodes of one level together. Each node--attribute interval is
divided into tiles of 2,048 entries, one CUDA block per tile. Within a level,
the device forms tile histograms, finds each attribute's best cut, and returns
these compact results to the host. The host applies the shared CART or C4.5
rule to choose the splits and finishes children that are leaves. The device
then stably partitions the columns and prepares histograms for the next level.
Each level therefore needs one round trip between device and host. Besides
choosing splits, the CPU stages transfers and runs the shared post-processing.
```
- Priority: high

### S-Opus-3
- Location: `approach.tex`, §\ImplSerialBase, "At each node it copies the row identifiers for every feature, ... so the work is limited by memory latency."
- Issue: terminology + style (split the long "therefore ... and ... so" sentence)
- Proposed rewrite:
```latex
\ImplSerialBase{} is a direct sequential implementation and the reference for
the others. At each node, it copies the sample identifiers for every
attribute, sorts them by value, scores every admissible cut, applies the
shared split-selection rule, and recurses on two new lists. For a node with
$n$ samples and $F$ attributes, sorting alone costs $O(Fn\log n)$. Values are
read indirectly through the sample identifiers, so the work is limited by
memory latency.
```
- Priority: high

### S-Opus-4
- Location: `approach.tex`, §\ImplSerialOpt, *Sort once*, "Every feature is sorted once before induction by a stable, single-threaded radix sort into 8-byte entries that pack a value with a row and class identifier. ... A feature that is constant on a node is dropped for its whole subtree."
- Issue: terminology + style (sentence reads ambiguously: "sorted ... by a sort into entries")
- Proposed rewrite:
```latex
\emph{Sort once.} Before induction, each attribute is converted into 8-byte
entries that pack a value with a sample and a class identifier, and these are
sorted once by a stable, single-threaded radix sort. A node occupies the same
index range in every column. After a split, the left samples are the first
$n_L$ entries of the winning column; the other columns are partitioned stably
to match, which keeps both children sorted, following the principle of SLIQ
and SPRINT~\cite{mehta1996sliq,shafer1996sprint}. An attribute that is
constant within a node is dropped for its whole subtree.
```
- Priority: high

### S-Opus-5
- Location: `approach.tex`, §\ImplParallel, "Nodes with at least 32,768 rows sweep features and partition columns in parallel. After a node with at least 4,096 rows is split, ..."
- Issue: terminology (+ "nodes sweep features" is an odd agent/action pairing)
- Proposed rewrite:
```latex
of Aldinucci et al.~\cite{aldinucci2014fastflow}. For nodes with at least
32,768 samples, attributes are swept and columns are partitioned in parallel.
After a node with at least 4,096 samples is split, one child becomes a pool
task while the current worker continues with the other.
```
- Priority: high

### S-Opus-6
- Location: `approach.tex`, §\ImplHybrid, "The threshold $m$ is configurable; by default it is $N/8192$, clamped to $[1024,2048]$, for CART and $N/4096$, clamped to $[1024,4096]$, for C4.5."
- Issue: style (nested commas make the two defaults hard to parse; unit of $m$ and meaning of $N$ not stated; "smaller children" is vague)
- Proposed rewrite:
```latex
\ImplHybrid{} retains level-synchronous GPU growth for large nodes but sends
children with fewer than $m$ samples to the \ImplParallel{} builder. The
threshold $m$ can be set explicitly. By default, it is $N/8192$ for CART and
$N/4096$ for C4.5, where $N$ is the number of training samples, clamped to
$[1024,2048]$ and $[1024,4096]$, respectively. These empirical defaults
reflect that a GPU level has substantial fixed overhead, whereas CPU threads
can grow small subtrees independently.
```
- Priority: med
- Note: "where $N$ is the number of training samples" is a definition, not a
  new claim; please confirm $N$ is not already defined elsewhere with a
  different symbol (Section 4 / dataset table).

### S-Opus-7
- Location: `approach.tex`, §\ImplHybrid, "The hand-off is asynchronous. A kernel gathers only columns that can still split the child into a device ring; copy streams move them through pinned buffers into a pageable host arena; a host callback queued behind each copy submits the CPU tasks. ... Assignment is therefore per node of one tree, not per tree of a forest, ..."
- Issue: style (semicolon chain; "split the child into a device ring" misparses) + terminology ("pinned" vs. "page-locked" used in §\ImplGpu)
- Proposed rewrite:
```latex
The hand-off is asynchronous. A kernel gathers the columns that can still
split the child into a ring buffer on the device. Copy streams move them
through page-locked buffers into a pageable host arena, and a host callback
queued behind each copy submits the CPU tasks. The GPU level loop does not
wait for these tasks. The arena is pre-touched by CPU threads but is not bound
to the GPU's NUMA node. Nodes from both devices enter one shared store and are
assembled in preorder after both finish. Devices are thus assigned individual
nodes of one tree, not whole trees of a forest, and the CPU grows subtrees
rather than only controlling the GPU.
```
- Priority: med

### S-Opus-8
- Location: `approach.tex`, §\ImplParallel, second paragraph, "The CPU path has no socket-aware scheduler and no explicit NUMA memory policy: ... so \texttt{taskset} or \texttt{numactl} can confine a run; page placement is otherwise left to Linux first-touch. Huge pages reduce translation overhead but do not control placement."
- Issue: shorten + flag-adjacent (tool names `taskset`/`numactl` are run-time details, in the same spirit as CLI flags; the affinity mask fact is enough)
- Proposed rewrite:
```latex
The CPU code has no socket-aware scheduler and no explicit NUMA memory policy:
workers are not pinned, and columns are not divided by socket. The thread pool
respects the process affinity mask, so a run can be confined to a subset of
cores. Pages are otherwise placed by the Linux first-touch policy. Huge pages
reduce address-translation overhead but do not control placement.
```
- Priority: med
- Note: if Section 4 says runs were confined with `numactl`, keep the tool name
  there (setup), not here (design).

### S-Opus-9
- Location: `approach.tex`, §\ImplGpu, second paragraph, "On the evaluated A100 and H100, cuts are scored directly in double precision. Devices with slow double precision may first apply a conservative single-precision filter and then score every surviving candidate exactly."
- Issue: shorten (the fallback path is not evaluated; one sentence suffices)
- Proposed rewrite:
```latex
The device stores the same 8-byte entries in one slot per attribute plus one
spare; CUB performs a stable radix presort as columns arrive. On the A100 and
H100 used here, cuts are scored directly in double precision. On devices with
slow double-precision arithmetic, a conservative single-precision filter can
discard cuts first, and the remaining ones are scored exactly.
```
- Priority: low

### S-Opus-10
- Location: `approach.tex`, §\ImplGpu, third paragraph, "Uploads pass through page-locked staging buffers rather than pinning the training matrix. This is the implementation's only explicit NUMA placement: ... General training memory and workers are not bound this way."
- Issue: style (slightly tighter; avoids "pinning" in two different senses, memory pinning vs. thread pinning)
- Proposed rewrite:
```latex
Uploads pass through page-locked staging buffers, so the training matrix
itself is not page-locked. These buffers are the only explicit NUMA placement
in the implementation: on a multi-socket host, they are allocated by a thread
bound to the GPU's NUMA node, and dedicated copy threads run on that node's
CPUs. Other training memory and the worker threads are not bound.
```
- Priority: low
- Note: original says "the GPU's PCI NUMA node"; "PCI" dropped as redundant
  (the NUMA node of a GPU is defined by its PCIe attachment). Restore if the
  authors want to be explicit.

### S-Opus-11
- Location: `related-work.tex`, "parallelised" (lines 33, 40), "organise" (line 78); `introduction.tex` uses "organized" (line 85)
- Issue: other (spelling consistency across the article; Section 3 uses none of these words)
- Proposed rewrite:
```latex
% related-work.tex
Tree induction can be parallelized in two ways.
% ...
et al.~\cite{aldinucci2014fastflow} parallelized YaDT on multicore processors with
% ...
CudaTree organize the work by node size, but only on the GPU. CudaTree with
```
- Priority: low (pick one variant; American matches the Introduction and the
  rest of Section 3, e.g. "Optimizations")

### S-Opus-12
- Location: `experiment-design.tex` lines 79–80, `\plan{./tree backends (\texttt{--simple}, \texttt{--serial}, \texttt{--parallel}, \texttt{--cuda}, \texttt{--gpu-only}); ...}`
- Issue: flag (outside Section 3; currently a `\plan{}` note, but it will become prose)
- Proposed rewrite: when this note is turned into text, refer to the
  implementations by their macros only, e.g.
```latex
We run \ImplSerialBase{}, \ImplSerialOpt{}, \ImplParallel{}, \ImplGpu{} and
\ImplHybrid{}, together with scikit-learn and ...
```
- Priority: low (reminder; not part of Section 3)

---

## Global find-replace table (rows → samples, features → attributes)

Search scope: all `paper/**/*.tex` except `sections/approach-planning/`.
Result: **only `approach.tex` contains "row(s)" or "feature(s)"**. The
Introduction, Related Work, Experiment Design, Results, Discussion and
Conclusion already use "samples" / "attributes" (or neither). No other section
needs changes. The planning drafts and diagrams under `approach-planning/`
also use "rows" (e.g. `diagram-final-grok.tex`: "at least \(m\) rows";
`diagram-final-muse.tex`: `--gpu-min-rows`); fix whichever diagram is adopted.

Do not do a blind global replace: "row" also appears in TikZ comments
("GPU row", "CPU row") meaning a diagram lane, and "column" stays correct
(columns are the per-attribute arrays). Replace per occurrence:

| File:line (approach.tex) | Current | Replacement | Covered by |
|---|---|---|---|
| 4 | Five configurations | Our five implementations | S-Opus-1 |
| 5 | a node's rows remain sorted | the samples of a node are kept sorted | S-Opus-1 |
| 6 | Feature values | Attribute values | S-Opus-1 |
| 7 | Every configuration | All implementations | S-Opus-1 |
| 10 | the lower feature index | the attribute with the lower index | S-Opus-1 |
| 14 | all five configurations | all five implementations | S-Opus-1 |
| 22 | row identifiers for every feature | sample identifiers for every attribute | S-Opus-3 |
| 24 | $n$ rows and $F$ features | $n$ samples and $F$ attributes | S-Opus-3 |
| 26 | through row identifiers | through the sample identifiers | S-Opus-3 |
| 33 | Every feature is sorted | each attribute is ... sorted | S-Opus-4 |
| 34 | a row and class identifier | a sample and a class identifier | S-Opus-4 |
| 39 | A feature that is constant on a node | An attribute that is constant within a node | S-Opus-4 |
| 59 | rows sweep features | samples, attributes are swept | S-Opus-5 |
| 60 | 4,096 rows | 4,096 samples | S-Opus-5 |
| 76 | to one row | to one sample | S-Opus-2 |
| 78 | node--feature interval | node--attribute interval | S-Opus-2 |
| 80 | feature's best cut | attribute's best cut | S-Opus-2 |
| 86 | one slot per feature | one slot per attribute | S-Opus-9 |

Related consistency items (same pass):

| Term | Use | Avoid | Where |
|---|---|---|---|
| page-locked | page-locked buffers | "pinned buffers" (line 110) | approach.tex, S-Opus-7 |
| implementations | the five systems | "configurations" | approach.tex, S-Opus-1 |
| -ize spelling | parallelized, organize | parallelised, organise | related-work.tex, S-Opus-11 |
| sample identifier | id of a training sample | "row identifier", "row id" | approach.tex, any diagram text |
