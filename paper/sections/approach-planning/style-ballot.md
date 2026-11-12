# Style ballot — Section 3 (+ terminology)

Vote YES only if the candidate is clearly better. Threshold for adoption: **3/4**.

Author rules: match Intro/RW style; no CLI flags; shorten low-value detail; **samples** not rows, **attributes** not features.

---

## B1 — Global terminology in `approach.tex`
Replace throughout approach.tex (careful phrasing):
- rows → samples; row → sample (incl. “row identifiers” → “sample identifiers”)
- features → attributes; feature → attribute (incl. “node--feature” → “node--attribute”, “A feature” → “An attribute”)

Intro/Related Work already use samples/attributes. No change needed there.

---

## B2 — Remove CLI flag from GPU subsection
Current: `\ImplGpu{} (\texttt{--cuda --gpu-only}) sets the hand-off threshold to one row,`
Candidate: `\ImplGpu{} sets the hand-off threshold to one sample,`

---

## B3 — Opening: shorter sentences + terminology
Replace the opening paragraph(s) of Section 3 with:

```
Five implementations grow the same exact CART or C4.5 tree. They differ in how
a node's samples remain sorted, which cuts are scored exactly, and where and in
what order nodes are grown. Attribute values are stored column-major as 32-bit
floats. Class labels are mapped to dense integers. Every implementation uses
the same CART and C4.5 rules and the same double-precision gain calculation,
compiled for both CPU and GPU. Gains within $10^{-12}$ are ties; the earlier
cut wins, then the attribute with the lower index. Optimizations may discard
only cuts that cannot change this decision. Concurrent builders place nodes in
a shared store and renumber them into flat preorder arrays at the end, so
creation order cannot alter the output. Our tests require identical trees from
all five implementations. When pruning is enabled, shared CPU code runs it
after growth. \ImplGpu{} and \ImplHybrid{} are two modes of one CUDA grower,
without and with hand-offs of small subtrees to the CPU.
```

---

## B4 — Serial Simple: terminology + split long sentence
Replace Serial Simple body with:

```
\ImplSerialBase{} is a direct sequential implementation and the reference for
the others. At each node it copies the sample identifiers for every attribute,
sorts them by value, scores every admissible cut, applies the shared
split-selection rule, and recurses on two new lists. A node with $n$ samples
and $F$ attributes therefore costs $O(Fn\log n)$ in sorting alone. Values are
read indirectly through sample identifiers, so the work is limited by memory
latency.
```

---

## B5 — Sort once: clearer wording + attributes
Replace the \emph{Sort once.} paragraph with:

```
\emph{Sort once.} Every attribute is sorted once before induction by a stable,
single-threaded radix sort. Each column holds 8-byte entries that pack a value
with a sample and class identifier. A node occupies the same index range in
every column. After a split, the left samples are the first $n_L$ entries of
the winning column; the other columns are partitioned stably to match, which
keeps both children sorted, following the principle of SLIQ and
SPRINT~\cite{mehta1996sliq,shafer1996sprint}. An attribute that is constant on
a node is dropped for its whole subtree.
```

---

## B6 — Parallel thresholds: samples/attributes + clearer syntax
Replace first Parallel paragraph with:

```
\ImplParallel{} combines two granularities, in the spirit of the NAP strategy
of Aldinucci et al.~\cite{aldinucci2014fastflow}. For nodes with at least
32,768 samples, attribute sweeps and column partitions run in parallel. After
a node with at least 4,096 samples is split, one child becomes a pool task
while the current worker continues with the other. Thus data parallelism
serves the few large nodes near the root, and task parallelism the many
smaller subtrees below. Presort and final preorder assembly are parallel as
well, and up to $\min(F-1,p)$ spare slots let $p$ threads move columns
concurrently.
```

---

## B7 — Shorten Hybrid hand-off (split semicolons; drop closing claim already in Intro/RW)
Replace Hybrid second paragraph with:

```
The hand-off is asynchronous. A kernel gathers only columns that can still
split the child into a device ring. Copy streams move them through page-locked
buffers into a pageable host arena. A host callback queued behind each copy
submits the CPU tasks. The GPU level loop does not wait for these tasks. The
arena is pre-touched by CPU threads but is not bound to the GPU's NUMA node.
Nodes from both devices enter one shared store and are assembled in preorder
after both finish.
```

(Drops the forest/control closing sentence as redundant with Intro/RW.)

---

## B8 — Shorten CPU NUMA paragraph (drop tool names)
Replace Parallel NUMA paragraph with:

```
The CPU path has no socket-aware scheduler and no explicit NUMA memory policy:
workers are not pinned and columns are not divided by socket. The pool
respects the process affinity mask; page placement is otherwise left to Linux
first-touch. Huge pages reduce translation overhead but do not control
placement.
```

---

## B9 — Shorten GPU host/CPU role sentence (remove repetition)
Current ends GPU ¶1 with: `The CPU still stages transfers, chooses splits, and runs shared post-processing.`
But host already “applies the shared … rule”. Candidate last sentence of that paragraph:
`Besides choosing splits, the CPU stages transfers and runs the shared post-processing.`

---

## B10 — Drop “PCI” from “GPU's PCI NUMA node” as redundant
Candidate: `the GPU's NUMA node` (keep rest of sentence).
