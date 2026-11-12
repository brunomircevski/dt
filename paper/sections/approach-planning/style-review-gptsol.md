# Section 3 language and terminology review

### S-GPT-1
- Location: `paper/sections/approach.tex`, opening paragraph, from “Five configurations grow the same exact CART or C4.5 tree.”
- Issue: style / shorten / terminology
- Proposed rewrite: exact LaTeX replacement

```latex
Five configurations grow the same exact CART or C4.5 tree. They differ in how
they keep a node's samples sorted, which cuts they score exactly, and how they
schedule nodes. Attribute values are stored column-major as 32-bit floats, and
class labels are mapped to dense integers. Every configuration uses the same
CART and C4.5 rules and the same double-precision gain calculation on the CPU
and GPU. Gains within $10^{-12}$ are ties; the earlier cut, followed by the
lower attribute index, wins. Optimizations may discard only cuts that cannot
change this decision. Concurrent builders place nodes in a shared store and
renumber them into flat preorder arrays at the end, so creation order cannot
alter the output. Our tests require identical trees from all five
configurations. When pruning is enabled, shared CPU code runs it after growth.
\ImplGpu{} and \ImplHybrid{} are two modes of one CUDA grower, without and
with hand-offs of small subtrees to the CPU.
```
- Priority: high

### S-GPT-2
- Location: `paper/sections/approach.tex`, `\ImplSerialBase{}`, from “At each node it copies the row identifiers”
- Issue: style / terminology / shorten
- Proposed rewrite: exact LaTeX replacement

```latex
\ImplSerialBase{} is a direct sequential implementation and the reference for
the others. At each node, it copies the sample identifiers for every
attribute, sorts them by value, scores every admissible cut, applies the shared
split-selection rule, and recurses on two new lists. For a node with $n$
samples and $F$ attributes, sorting alone costs $O(Fn\log n)$. Indirect value
access through sample identifiers also makes the implementation sensitive to
memory latency.
```
- Priority: high

### S-GPT-3
- Location: `paper/sections/approach.tex`, `\emph{Sort once.}`, from “Every feature is sorted once”
- Issue: terminology / style
- Proposed rewrite: exact LaTeX replacement

```latex
\emph{Sort once.} Every attribute is sorted once before induction by a stable,
single-threaded radix sort into 8-byte entries that pack a value with a sample
and class identifier. A node occupies the same index range in every column.
After a split, the left samples are the first $n_L$ entries of the winning
column. The other columns are partitioned stably to match, which keeps both
children sorted, following the principle of SLIQ and
SPRINT~\cite{mehta1996sliq,shafer1996sprint}. An attribute that is constant at
a node is dropped for its whole subtree.
```
- Priority: high

### S-GPT-4
- Location: `paper/sections/approach.tex`, `\emph{Score few cuts exactly.}`, from “The sweep updates class counts”
- Issue: style / shorten
- Proposed rewrite: exact LaTeX replacement

```latex
\emph{Score few cuts exactly.} The sweep updates class counts and computes a
cheap gain estimate with bounded rounding error. Only candidates that can
still beat the incumbent under conservative bounds are scored by the shared
routine, in cut order. For two classes, convexity allows whole blocks of cuts
to be skipped. The winning sweep also provides the class counts of both
children.
```
- Priority: med

### S-GPT-5
- Location: `paper/sections/approach.tex`, `\emph{Allocate once.}`, from “Columns live in slots of fixed capacity.”
- Issue: style / shorten
- Proposed rewrite: exact LaTeX replacement

```latex
\emph{Allocate once.} Columns occupy fixed-capacity slots. The serial builder
keeps one spare slot: large ranges are partitioned out of place, while smaller
ranges reuse a buffer. Large allocations request transparent huge pages, and
an explicit stack replaces recursion. \ImplParallel{} uses the same builder
with a thread pool.
```
- Priority: med

### S-GPT-6
- Location: `paper/sections/approach.tex`, `\ImplParallel{}`, from “Nodes with at least 32,768 rows”
- Issue: terminology / style
- Proposed rewrite: exact LaTeX replacement

```latex
\ImplParallel{} combines two granularities, in the spirit of the NAP strategy
of Aldinucci et al.~\cite{aldinucci2014fastflow}. For nodes with at least
32,768 samples, it scans attributes and partitions columns in parallel. After
a node with at least 4,096 samples is split, one child becomes a pool task
while the current worker continues with the other. Data parallelism therefore
serves the few large nodes near the root, while task parallelism serves the
many smaller subtrees. Presorting and final preorder assembly are also
parallel, and up to $\min(F-1,p)$ spare slots let $p$ threads move columns
concurrently.
```
- Priority: high

### S-GPT-7
- Location: `paper/sections/approach.tex`, end of `\ImplParallel{}`, from “The CPU path has no socket-aware scheduler”
- Issue: shorten / style
- Proposed rewrite: exact LaTeX replacement

```latex
The CPU path has no socket-aware scheduler or explicit NUMA memory policy:
workers are not pinned, and columns are not assigned by socket. The pool
respects the process affinity mask, but Linux first-touch otherwise determines
page placement. Huge pages reduce translation overhead but do not control
placement.
```
- Priority: med

### S-GPT-8
- Location: `paper/sections/approach.tex`, first sentence of `\ImplGpu{}`, “\ImplGpu{} (\texttt{--cuda --gpu-only}) sets the hand-off threshold to one row”
- Issue: flag / terminology
- Proposed rewrite: exact LaTeX replacement

```latex
\ImplGpu{} sets the hand-off threshold to one sample, so every nonempty
subtree remains on the GPU.
```
- Priority: high

### S-GPT-9
- Location: `paper/sections/approach.tex`, `\ImplGpu{}`, from “Unlike the depth-first CPU builders”
- Issue: terminology / style / shorten
- Proposed rewrite: exact LaTeX replacement

```latex
Unlike the depth-first CPU builders, \ImplGpu{} processes all active nodes at
one level together. Each node--attribute interval is divided into tiles of
2,048 entries, with one CUDA block per tile. At each level, the device forms
tile histograms, finds the best cut for each attribute, and returns these
compact results to the host. The host applies the shared CART or C4.5 choice
rule and finishes children that are leaves. The device then stably partitions
the columns and prepares histograms for the next level. This requires one
split-decision round trip per level. The CPU also stages transfers and runs
shared post-processing.
```
- Priority: high

### S-GPT-10
- Location: `paper/sections/approach.tex`, `\ImplGpu{}`, from “The device stores the same 8-byte entries”
- Issue: terminology / style
- Proposed rewrite: exact LaTeX replacement

```latex
The device stores the same 8-byte entries in one slot per attribute, plus one
spare slot. CUB performs a stable radix presort as the columns arrive. On the
evaluated A100 and H100 GPUs, cuts are scored directly in double precision.
Devices with slow double precision may first apply a conservative
single-precision filter and then score every surviving candidate exactly.
```
- Priority: med

### S-GPT-11
- Location: `paper/sections/approach.tex`, `\ImplHybrid{}`, from “The threshold $m$ is configurable”
- Issue: style / shorten
- Proposed rewrite: exact LaTeX replacement

```latex
\ImplHybrid{} retains level-synchronous GPU growth for large nodes but sends
smaller children to the \ImplParallel{} builder. The hand-off threshold $m$ is
configurable. By default, it is $N/8192$, clamped to 1,024--2,048 samples, for
CART and $N/4096$, clamped to 1,024--4,096 samples, for C4.5. These empirical
defaults account for the fixed overhead of a GPU level and the CPU's ability
to grow small subtrees independently.
```
- Priority: med

### S-GPT-12
- Location: `paper/sections/experiment-design.tex`, `\subsection{Compared Implementations}`, “./tree backends (\texttt{--simple}, \texttt{--serial}, \texttt{--parallel}, \texttt{--cuda}, \texttt{--gpu-only})”
- Issue: flag / other
- Proposed rewrite: exact LaTeX replacement

```latex
\plan{The five implementations in this work; scikit-learn and Weka J48 on the
CPU; cuML and XGBoost on the GPU, with their settings. Explain why each is the
closest equivalent (no public GPU C4.5; J48 depth limit adapter; XGBoost
squared error = Gini on bins). Add rpart and YaDT, or explain their omission.}
```
- Priority: high

## Global terminology find--replace

Apply terminology changes to prose, captions, and comments that may become
prose. Review each match rather than using an unconditional replacement,
because compounds and grammar require small adjustments.

| Scope | Find | Replace | Current status / exceptions |
|---|---|---|---|
| `introduction.tex` | `row`, `rows` | `sample`, `samples` | No matches. |
| `introduction.tex` | `feature`, `features` | `attribute`, `attributes` | No matches. |
| `related-work.tex` | `row`, `rows` | `sample`, `samples` | No matches. |
| `related-work.tex` | `feature`, `features` | `attribute`, `attributes` | No matches. |
| `approach.tex` | `row identifiers` | `sample identifiers` | Replace all prose occurrences. |
| `approach.tex` | `row`, `rows` | `sample`, `samples` | Replace all eight occurrences; preserve number and compounds. |
| `approach.tex` | `feature index` | `attribute index` | Replace. |
| `approach.tex` | `node--feature interval` | `node--attribute interval` | Replace. |
| `approach.tex` | `feature`, `features`, `feature's` | `attribute`, `attributes`, `attribute's` | Replace all ten occurrences, preserving possessives and number. |

No `row(s)` or `feature(s)` matches occur in the current main text of
`experiment-design.tex`, `results.tex`, `discussion.tex`, or `conclusion.tex`.
The files under `approach-planning/` contain draft-only matches and need not be
normalized unless one of them is promoted into the paper. Separately,
`experiment-design.tex` still names command-line flags in a planning note;
S-GPT-12 removes them.
