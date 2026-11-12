# Section 3 — Style and Terminology Review (Grok)

Scope: `paper/sections/approach.tex`, read against `introduction.tex` and
`related-work.tex`. No paper files were edited.

Introduction and related work already use short sentences and say only
*samples* and *attributes*. Section 3 is the outlier: it says *rows* and
*features*, names CLI flags, and joins unrelated facts into long sentences.
The proposals below are surgical. Each rewrite keeps the original facts.

Apply the numbered proposals, not a second blind pass over the same spans.
The table is the checklist of every *rows* / *features* occurrence, for use
where a proposal does not already rewrite that sentence.

Left as local technical words: *columns* (storage layout), *sweep* (the scan
already named in this section), *slots*, *builders*, and *configurations*.
*Configurations* is narrower than the introduction's *five implementations*:
here it names five run settings of fewer code paths (one serial builder with
or without a thread pool; one CUDA grower with or without hand-off). Not
proposed as a replace.

## Global find–replace table

Safe order if applying the table alone: longer phrases and the article change
(`A feature` → `An attribute`) before the bare words. Do not replace inside
text a proposal already rewrites.

| Find | Replace | Where | Notes |
|---|---|---|---|
| `a node's rows` | `a node's samples` | `approach.tex` opening | |
| `row identifiers` | `sample identifiers` | `approach.tex`, serial baseline (twice) | Prose, not a code idiom |
| `a row and class identifier` | `a sample identifier and a class identifier` | `approach.tex`, *Sort once* | Unpacks the packed entry; same contents |
| `$n$ rows` | `$n$ samples` | `approach.tex`, serial baseline | `$n$` unchanged |
| `32,768 rows` | `32,768 samples` | `approach.tex`, parallel | Node-size threshold |
| `4,096 rows` | `4,096 samples` | `approach.tex`, parallel | Node-size threshold |
| `one row` | `one sample` | `approach.tex`, GPU hand-off | Only this occurrence |
| `Feature values` | `Attribute values` | `approach.tex` opening | Sentence-initial |
| `feature index` | `attribute index` | `approach.tex` opening | Tie-break; not a code name |
| `every feature` | `every attribute` | `approach.tex`, serial baseline | |
| `Every feature` | `Every attribute` | `approach.tex`, *Sort once* | |
| `$F$ features` | `$F$ attributes` | `approach.tex`, serial baseline | `$F$` unchanged |
| `sweep features` | `sweep attributes` | `approach.tex`, parallel | |
| `A feature that` | `An attribute that` | `approach.tex`, *Sort once* | Article must change with the noun |
| `node--feature` | `node--attribute` | `approach.tex`, GPU | LaTeX en-dash |
| `feature's` | `attribute's` | `approach.tex`, GPU | Or the S-Grok-5 recast |
| `per feature` | `per attribute` | `approach.tex`, GPU slots | |
| `rows` / `features` | — none — | `introduction.tex`, `related-work.tex` | Already *samples* / *attributes* only |
| `rows` / `features` | — none — | `experiment-design.tex`, `results.tex`, `discussion.tex`, `conclusion.tex`, `main.tex`, `chart-tests.tex` | No replace. `experiment-design.tex` names `--simple`, `--serial`, `--parallel`, `--cuda`, and `--gpu-only` inside a `\plan` note (Compared Implementations). If that note becomes prose, drop the flags |
| `rows` / `features` in `approach-planning/` | leave | Drafts, votes, TikZ | Not article text. Diagram comments use *row* for a picture row (`GPU row`, `Transfer row`); do not bulk-replace |

No code or math idiom in Section 3 requires *row* or *feature*. `$n$` is the
sample count and `$F$` the attribute count either way.

---

### S-Grok-1
- Location: `approach.tex`, opening paragraph — "They differ in how a node's rows remain sorted" through "renumber them into flat preorder arrays at the end, so creation order cannot alter the output."
- Issue: terminology / style
- Proposed rewrite:
```latex
Five configurations grow the same exact CART or C4.5 tree. They differ in how
a node's samples remain sorted, which cuts are scored exactly, and where and in
what order nodes are grown. Attribute values are stored column-major as 32-bit
floats. Class labels are mapped to dense integers. Every configuration
uses the same CART and C4.5 rules and the same double-precision gain
calculation, compiled for both CPU and GPU. Gains within $10^{-12}$ are ties;
the earlier cut, then the lower attribute index, wins. Optimizations may discard
only cuts that cannot change this decision. Concurrent builders place nodes in
a shared store. They renumber them into flat preorder arrays at the end, so
creation order cannot alter the output. Our tests require identical trees from
all five configurations. When pruning is enabled, shared CPU code runs it after
growth. \ImplGpu{} and \ImplHybrid{} are two modes of one CUDA grower, without
and with hand-offs of small subtrees to the CPU.
```
- Priority: high

### S-Grok-2
- Location: `approach.tex`, `\subsection{\ImplSerialBase}` — "At each node it copies the row identifiers for every feature, sorts them by value, scores every admissible cut, applies the shared split-selection rule, and recurses on two new lists. A node with $n$ rows and $F$ features therefore costs $O(Fn\log n)$ in sorting alone, and values are read indirectly through row identifiers, so the work is limited by memory latency."
- Issue: terminology / style
- Proposed rewrite:
```latex
\ImplSerialBase{} is a direct sequential implementation and the reference for
the others. At each node it copies the sample identifiers for every attribute
and sorts them by value. It then scores every admissible cut, applies the
shared split-selection rule, and recurses on two new lists. A node with $n$
samples and $F$ attributes therefore costs $O(Fn\log n)$ in sorting alone.
Values are read indirectly through sample identifiers, so the work is limited
by memory latency.
```
- Priority: high

### S-Grok-3
- Location: `approach.tex`, `\emph{Sort once.}` — "Every feature is sorted once before induction by a stable, single-threaded radix sort into 8-byte entries that pack a value with a row and class identifier." through "A feature that is constant on a node is dropped for its whole subtree."
- Issue: terminology / style
- Proposed rewrite:
```latex
\emph{Sort once.} Every attribute is sorted once before induction by a stable,
single-threaded radix sort. Each entry is 8 bytes and packs a value with a
sample identifier and a class identifier. A node occupies the same index range
in every column. After a split, the left samples are the first $n_L$ entries
of the winning column. The other columns are partitioned stably to match, which
keeps both children sorted, following the principle of SLIQ and
SPRINT~\cite{mehta1996sliq,shafer1996sprint}. An attribute that is constant on
a node is dropped for its whole subtree.
```
- Priority: high

### S-Grok-4
- Location: `approach.tex`, `\subsection{\ImplParallel}` — "Nodes with at least 32,768 rows sweep features and partition columns in parallel. After a node with at least 4,096 rows is split" through "up to $\min(F-1,p)$ spare slots let $p$ threads move columns concurrently."
- Issue: terminology / style
- Proposed rewrite:
```latex
\ImplParallel{} combines two granularities, in the spirit of the NAP strategy
of Aldinucci et al.~\cite{aldinucci2014fastflow}. Nodes with at least 32,768
samples sweep attributes and partition columns in parallel. After a node with
at least 4,096 samples is split, one child becomes a pool task while the
current worker continues with the other. Thus data parallelism serves the few
large nodes near the root, and task parallelism the many smaller subtrees
below. Presort and final preorder assembly are parallel as well. Up to
$\min(F-1,p)$ spare slots let $p$ threads move columns concurrently.
```
- Priority: high

### S-Grok-5
- Location: `approach.tex`, `\subsection{\ImplGpu}` — "\ImplGpu{} (\texttt{--cuda --gpu-only}) sets the hand-off threshold to one row" through "finishes children that are leaves."
- Issue: flag / terminology / style
- Proposed rewrite:
```latex
\ImplGpu{} sets the hand-off threshold to one sample, so every nonempty
subtree remains on the GPU. Unlike the depth-first CPU builders, it processes
all active nodes of one level together. Each node--attribute interval is
divided into tiles of 2,048 entries, one CUDA block per tile. Within a level,
the device forms tile histograms and finds the best cut of each attribute.
It returns these compact results to the host. The host applies the shared
CART or C4.5 choice rule and finishes children that are leaves. The device
then stably partitions the columns and prepares histograms for the next level.
This is one split-decision round trip per level. The CPU still stages
transfers, chooses splits, and runs shared post-processing.
```
- Priority: high

### S-Grok-6
- Location: `approach.tex`, `\subsection{\ImplHybrid}` — "A kernel gathers only columns that can still split the child into a device ring; copy streams move them through pinned buffers into a pageable host arena; a host callback queued behind each copy submits the CPU tasks." through "the CPU grows subtrees rather than only controlling the GPU."
- Issue: style / shorten
- Proposed rewrite:
```latex
The hand-off is asynchronous. A kernel gathers only the columns that can still
split the child into a device ring. Copy streams move them through pinned
buffers into a pageable host arena. A host callback queued behind each copy
submits the CPU tasks. The GPU level loop does not wait for these tasks. The
arena is pre-touched by CPU threads but is not bound to the GPU's NUMA node.
Nodes from both devices enter one shared store and are assembled in preorder
after both finish.
```
- Priority: high

The dropped closing sentence restates the introduction (nodes of one tree
assigned by size; the CPU grows part of the tree) and the last paragraph of
related work (per node, not per tree of a forest). The hand-off sentences
above already say what each device does.

### S-Grok-7
- Location: `approach.tex`, `\subsection{\ImplParallel}` — "The CPU path has no socket-aware scheduler and no explicit NUMA memory policy: workers are not pinned and columns are not divided by socket. The pool respects the process affinity mask, so \texttt{taskset} or \texttt{numactl} can confine a run; page placement is otherwise left to Linux first-touch."
- Issue: shorten / style
- Proposed rewrite:
```latex
The CPU path has no socket-aware scheduler and no explicit NUMA memory policy.
Workers are not pinned, and columns are not divided by socket. The pool
respects the process affinity mask. Page placement is left to Linux
first-touch. Huge pages reduce translation overhead but do not control
placement.
```
- Priority: med

Drops the `taskset` / `numactl` invocation. Affinity, first-touch, and the
huge-page limit stay. Huge pages are still introduced under *Allocate once*.

### S-Grok-8
- Location: `approach.tex`, `\subsection{\ImplGpu}` — "The device stores the same 8-byte entries in one slot per feature plus one spare; CUB performs a stable radix presort as columns arrive." through "General training memory and workers are not bound this way."
- Issue: terminology / style
- Proposed rewrite:
```latex
The device stores the same 8-byte entries in one slot per attribute plus one
spare. CUB performs a stable radix presort as columns arrive. On the evaluated
A100 and H100, cuts are scored directly in double precision. Devices with slow
double precision may first apply a conservative single-precision filter.
Every surviving candidate is then scored exactly.

Uploads pass through page-locked staging buffers rather than pinning the
training matrix. This is the implementation's only explicit NUMA placement.
On a multi-socket host, the buffers are allocated by a thread bound to the
GPU's PCI NUMA node. Dedicated copy threads are pinned to that node's CPUs.
General training memory and workers are not bound this way.
```
- Priority: med

### S-Grok-9
- Location: `approach.tex`, `\emph{Allocate once.}` — "Columns live in slots of fixed capacity. The serial builder keeps one spare slot so large ranges can be partitioned out of place; smaller ranges reuse a buffer. Large allocations request transparent huge pages, and an explicit stack replaces recursion."
- Issue: style
- Proposed rewrite:
```latex
\emph{Allocate once.} Columns are stored in slots of fixed capacity. The serial
builder keeps one spare slot, so large ranges can be partitioned out of place.
Smaller ranges reuse a buffer. Large allocations request transparent huge
pages. An explicit stack replaces recursion. \ImplParallel{} is this same
builder with a thread pool enabled.
```
- Priority: med

### S-Grok-10
- Location: `approach.tex`, `\emph{Score few cuts exactly.}` — "The sweep updates class counts and a cheap gain estimate with a bounded rounding error. Only candidates that can still beat the incumbent under conservative bounds are scored by the shared routine, in cut order. For two classes, convexity lets whole blocks of cuts be skipped."
- Issue: style
- Proposed rewrite:
```latex
\emph{Score few cuts exactly.} The sweep updates class counts and a lower-cost
gain estimate whose rounding error is bounded. Only candidates that can still
beat the incumbent under conservative bounds are scored by the shared routine,
in cut order. For two classes, convexity allows whole blocks of cuts to be
skipped. The winning sweep also supplies the children's class counts.
```
- Priority: low

### S-Grok-11
- Location: `approach.tex`, `\subsection{\ImplHybrid}` — "The threshold $m$ is configurable; by default it is $N/8192$, clamped to $[1024,2048]$, for CART and $N/4096$, clamped to $[1024,4096]$, for C4.5."
- Issue: style / terminology
- Proposed rewrite:
```latex
\ImplHybrid{} retains level-synchronous GPU growth for large nodes but sends
smaller children to the \ImplParallel{} builder. The threshold $m$ is
configurable. By default it is $N/8192$, clamped to $[1024,2048]$ samples,
for CART, and $N/4096$, clamped to $[1024,4096]$ samples, for C4.5. These
empirical defaults reflect that a GPU level has substantial fixed overhead,
whereas CPU threads can grow small subtrees independently.
```
- Priority: low

The added word *samples* names the unit of the existing clamps. The fractions
and bounds are unchanged.
