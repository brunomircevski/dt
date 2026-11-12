# Section 3 — Style and Terminology Review (Muse Wonder)

Scope: `approach.tex` (primary), checked against `introduction.tex` and
`related-work.tex`. No paper files were edited. Intro/Related Work already
use short, direct sentences and consistently say *samples* / *attributes*;
Section 3 is the outlier on both counts.

## Global find–replace table (whole article)

Apply in prose only. Keep the symbols `$n$`, `$F$`, `$m$`, `$N$` unchanged.

| Find | Replace | Where it occurs | Notes |
|---|---|---|---|
| `a node's rows` | `a node's samples` | `approach.tex` only | Intro/Related Work: no occurrences |
| `Feature values` (sentence-initial) | `Attribute values` | `approach.tex` only | |
| `feature values` | `attribute values` | `approach.tex` only (if any remain after S-Muse-1) | |
| `the lower feature index` | `the lower attribute index` | `approach.tex` only | Tie rule; not a code idiom, safe to change |
| `for every feature` / `Every feature` | `for every attribute` / `Every attribute` | `approach.tex` only | |
| `A feature that is constant` | `An attribute that is constant` | `approach.tex` only | Watch the article `A` → `An` |
| `row identifiers` / `row identifier` | `sample identifiers` / `sample identifier` | `approach.tex` only | Prose, not code; safe to change |
| `a row and class identifier` | `a sample and class identifier` | `approach.tex` only | Packed-entry description |
| `$n$ rows and $F$ features` | `$n$ samples and $F$ attributes` | `approach.tex` only | Defines `$F$` as attribute count once |
| `32,768 rows` / `4,096 rows` (thresholds) | `32,768 samples` / `4,096 samples` | `approach.tex` only | |
| `sweep features` | `sweep attributes` (see S-Muse-4 rewrite) | `approach.tex` only | Better recast than literal replace |
| `one row` (hand-off threshold) | `one sample` | `approach.tex` only | |
| `node--feature interval` | `node--attribute interval` | `approach.tex` only | LaTeX en-dash compound |
| `feature's best cut` / `each feature's best cut` | `best cut for each attribute` | `approach.tex` only | Recast avoids possessive |
| `one slot per feature` | `one slot per attribute` | `approach.tex` only | |
| `rows` / `features` in `introduction.tex`, `related-work.tex` | — none found — | Both files already use only `samples` / `attributes` | No changes needed; do not touch |
| `rows` / `features` in `experiment-design.tex`, `results.tex`, `discussion.tex`, `conclusion.tex` | — none found — | Verified by search | No changes needed. Re-check after any future edit of those files |
| `rows` / `features` in `approach-planning/*` | leave as-is | Planning drafts, diagrams, votes | Not paper text; excluded from the article-wide rule |
| code/math idioms to keep | — none in Section 3 prose — | e.g. identifiers inside `\texttt{...}` would stay, but the only `\texttt{}` in Section 3 is the CLI flag removed by S-Muse-5 | No exception needed in the current text |

---

### S-Muse-1
- Location: `approach.tex`, opening paragraph — "They differ in how a node's rows remain sorted, which cuts are scored exactly, and where and in what order nodes are grown. Feature values are stored column-major as 32-bit floats, and class labels are mapped to dense integers. … Gains within $10^{-12}$ are ties; the earlier cut, then the lower feature index, wins."
- Issue: terminology / style
- Proposed rewrite:
```latex
They differ in how a node's samples remain sorted, which cuts are scored
exactly, and where and in what order nodes are grown. Attribute values are
stored column-major as 32-bit floats. Class labels are mapped to dense
integers. Every configuration uses the same CART and C4.5 rules and the same
double-precision gain calculation, compiled for both CPU and GPU. Gains
within $10^{-12}$ are ties; the earlier cut, then the lower attribute index,
wins.
```
- Priority: high

### S-Muse-2
- Location: `approach.tex`, `\subsection{\ImplSerialBase}` — "At each node it copies the row identifiers for every feature, sorts them by value, scores every admissible cut, applies the shared split-selection rule, and recurses on two new lists. A node with $n$ rows and $F$ features therefore costs $O(Fn\log n)$ in sorting alone, and values are read indirectly through row identifiers, so the work is limited by memory latency."
- Issue: terminology / style
- Proposed rewrite:
```latex
At each node it copies the sample identifiers for every attribute, sorts
them by value, scores every admissible cut, applies the shared
split-selection rule, and recurses on two new lists. A node with $n$ samples
and $F$ attributes therefore costs $O(Fn\log n)$ in sorting alone. Values
are read indirectly through sample identifiers, so the work is limited by
memory latency.
```
- Priority: high

### S-Muse-3
- Location: `approach.tex`, `\emph{Sort once.}` — "Every feature is sorted once before induction by a stable, single-threaded radix sort into 8-byte entries that pack a value with a row and class identifier. … A feature that is constant on a node is dropped for its whole subtree."
- Issue: terminology
- Proposed rewrite:
```latex
\emph{Sort once.} Every attribute is sorted once before induction by a
stable, single-threaded radix sort into 8-byte entries that pack a value
with a sample and class identifier. A node occupies the same index range in
every column. After a split, the left samples are the first $n_L$ entries of
the winning column; the other columns are partitioned stably to match, which
keeps both children sorted, following the principle of SLIQ and
SPRINT~\cite{mehta1996sliq,shafer1996sprint}. An attribute that is constant
on a node is dropped for its whole subtree.
```
- Priority: high

### S-Muse-4
- Location: `approach.tex`, `\subsection{\ImplParallel}` — "Nodes with at least 32,768 rows sweep features and partition columns in parallel. After a node with at least 4,096 rows is split, one child becomes a pool task while the current worker continues with the other."
- Issue: terminology / style
- Proposed rewrite:
```latex
For nodes with at least 32,768 samples, attribute sweeps and column
partitions run in parallel. After a node with at least 4,096 samples is
split, one child becomes a pool task while the current worker continues with
the other.
```
- Priority: high

### S-Muse-5
- Location: `approach.tex`, `\subsection{\ImplGpu}` — "\ImplGpu{} (\texttt{--cuda --gpu-only}) sets the hand-off threshold to one row, so every nonempty subtree remains on the GPU."
- Issue: flag / terminology
- Proposed rewrite:
```latex
\ImplGpu{} sets the hand-off threshold to one sample, so every nonempty
subtree remains on the GPU.
```
- Priority: high

### S-Muse-6
- Location: `approach.tex`, `\subsection{\ImplGpu}` — "Each node--feature interval is divided into tiles of 2,048 entries, one CUDA block per tile. Within a level the device forms tile histograms, finds each feature's best cut, and returns these compact results to the host, which applies the shared CART or C4.5 choice rule and finishes children that are leaves."
- Issue: terminology / style
- Proposed rewrite:
```latex
Each node--attribute interval is divided into tiles of 2,048 entries, with
one CUDA block per tile. Within a level, the device forms tile histograms,
finds the best cut for each attribute, and returns these compact results to
the host. The host applies the shared CART or C4.5 choice rule and finishes
children that are leaves.
```
- Priority: high

### S-Muse-7
- Location: `approach.tex`, `\subsection{\ImplGpu}` — "The device stores the same 8-byte entries in one slot per feature plus one spare; CUB performs a stable radix presort as columns arrive."
- Issue: terminology / style
- Proposed rewrite:
```latex
The device stores the same 8-byte entries in one slot per attribute plus one
spare. CUB performs a stable radix presort as columns arrive.
```
- Priority: med

### S-Muse-8
- Location: `approach.tex`, `\emph{Allocate once.}` — "Large allocations request transparent huge pages, and an explicit stack replaces recursion."
- Issue: shorten
- Proposed rewrite:
```latex
Columns live in slots of fixed capacity. The serial builder keeps one spare
slot so large ranges can be partitioned out of place; smaller ranges reuse a
buffer. An explicit stack replaces recursion.
```
- Priority: low

### S-Muse-9
- Location: `approach.tex`, `\subsection{\ImplParallel}`, closing paragraph — "The CPU path has no socket-aware scheduler and no explicit NUMA memory policy: workers are not pinned and columns are not divided by socket. The pool respects the process affinity mask, so \texttt{taskset} or \texttt{numactl} can confine a run; page placement is otherwise left to Linux first-touch. Huge pages reduce translation overhead but do not control placement."
- Issue: shorten / style
- Proposed rewrite:
```latex
The CPU path has no socket-aware scheduler and no explicit NUMA memory
policy. Workers are not pinned and columns are not divided by socket. The
pool respects the process affinity mask, and page placement is otherwise
left to Linux first-touch.
```
- Priority: med

### S-Muse-10
- Location: `approach.tex`, `\subsection{\ImplGpu}`, closing paragraph — "This is the implementation's only explicit NUMA placement: on a multi-socket host, the buffers are allocated by a thread bound to the GPU's PCI NUMA node, and dedicated copy threads are pinned to that node's CPUs. General training memory and workers are not bound this way."
- Issue: shorten
- Proposed rewrite:
```latex
This is the implementation's only explicit NUMA placement: on a multi-socket
host, the buffers are allocated by a thread bound to the GPU's PCI NUMA
node, and dedicated copy threads are pinned to that node's CPUs.
```
- Priority: med

### S-Muse-11
- Location: `approach.tex`, `\subsection{\ImplHybrid}` — "The threshold $m$ is configurable; by default it is $N/8192$, clamped to $[1024,2048]$, for CART and $N/4096$, clamped to $[1024,4096]$, for C4.5."
- Issue: style
- Proposed rewrite:
```latex
The threshold $m$ is configurable. By default, it is $N/8192$, clamped to
$[1024,2048]$, for CART and $N/4096$, clamped to $[1024,4096]$, for C4.5.
```
- Priority: low

### S-Muse-12
- Location: `approach.tex`, `\subsection{\ImplHybrid}` — "A kernel gathers only columns that can still split the child into a device ring; copy streams move them through pinned buffers into a pageable host arena; a host callback queued behind each copy submits the CPU tasks."
- Issue: style
- Proposed rewrite:
```latex
A kernel gathers only columns that can still split the child into a device
ring. Copy streams move them through pinned buffers into a pageable host
arena. A host callback queued behind each copy then submits the CPU tasks.
```
- Priority: med
