# Round 4 review (Opus)

## 1. Draft scores (1–10)

| Draft | Factual accuracy | Clarity | Professionalism | Completeness vs 2-page plan | Length | Mean |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT-Sol | 9 | 8 | 9 | 9 | 8 | 8.6 |
| Muse | 8 | 8 | 8 | 9 | 8 | 8.2 |
| Grok | 9 | 6 | 6 | 8 | 9 | 7.6 |

### GPT-Sol (1,100 words of text)

The most polished register of the three, and its claims are accurate.
Strong points:

* the "data parallelism near the root … task parallelism deeper" sentence;
* a precise NUMA statement: the staging buffers are allocated by a thread
  bound to the GPU's node, and workers and training memory are not bound;
* "huge pages … do not control placement";
* a closing sentence that sets the hybrid apart from earlier schemes.

Weak points:

* Some passages are abstract ("exposing parallelism across nodes,
  features, and values") and repeat ideas.
* The figure caption is too short to stand alone.

### Muse (1,080 words)

Good coverage, including smaller-side bitmap marking and the host
finishing leaf children. Problems:

* `$leftCount$` is typeset as math, which renders as italic juxtaposed
  letters.
* "Lower marginal cost" over-reads a size heuristic.
* "Checked against the verified trees of Sect. 1" points to a section that
  does not define them.

### Grok (1,036 words)

The most careful with facts. It notes that the NUMA pinning applies only
when the process may run off the GPU's node, and that each row is
transferred at most once. However:

* It is written one sentence per line, in short choppy sentences, which
  reads as notes rather than LNCS prose.
* It hard-codes the subsection titles instead of using the macros.
* "Rows that share a value may change order" is confusing in a paragraph
  about a stable sort.

## 2. Steals adopted in `final-candidate-opus.tex`

1. **GPT-Sol:** "Huge pages reduce translation overhead but do not
   control placement" (§3.4).
2. **Muse and GPT-Sol:** the host "finishes children that are leaves"
   during its one round trip (§3.5).
3. **Muse:** "without and with hand-offs of small subtrees to the CPU" in
   the lead-in.
4. **Grok and Muse:** the CPU's role in \ImplGpu{}: it stages copies,
   applies the split rule, and prunes for C4.5 (§3.5).
5. **GPT-Sol's** "data near the root, tasks deeper" sentence and **Muse's**
   "neither forest split nor CPU control" positioning sentence. I tried
   both and cut them for length; Related Work already makes the second
   point.

## 3. Diagram scores (clarity / value / fact-safety / buildability, each /10)

| Proposal | Clarity | Value | Fact-safety | Buildability | Note |
|---|:-:|:-:|:-:|:-:|---|
| Opus (round 3) | 7 | 7 | 9 | 9 | Working TikZ and shows the host round trip, but neither the overlap with the next level nor the size fork |
| GPT-Sol | 8 | 8 | 8 | 5 | Good swim lanes and time axis; NUMA labels in the figure risk being misread; ASCII only |
| Muse | 7 | 9 | 7 | 7 | Best on concurrency and the explicit size fork; draws buffer sizes (512 MiB, 2×128 MiB) and example node sizes the consensus omits; dense |
| Grok | 8 | 7 | 9 | 5 | Simple and safe; no host round trip or time axis; Mermaid only |

### Consensus design (implemented in `diagram-final-opus.tex`)

**Layout.** Three lanes (GPU, Host, CPU) with time running left to right:

* GPU lane: level \(d\), then a fork "child \(<m\)?":
  * no: level \(d{+}1\), then "…";
  * yes: down to the hand-off.
* Host lane: "pinned buffers → host arena", reached by an arrow labeled
  "gather into device ring, copy asynchronously".
* CPU lane: "subtree tasks", reached by a dashed host-callback arrow and
  aligned under level \(d{+}1\). Dotted guides mark it as "concurrent".
* Both lanes feed "shared node store → preorder tree".

The host's split decision appears inside the level boxes as "sweep →
host picks splits → partition", rather than as an extra loop of arrows.

**Omitted:**

* kernel names and buffer sizes;
* NUMA: no sockets drawn, and no NUMA label on the arena;
* the presort and upload;
* the float filter;
* timings;
* the \(m\) formula.

**\ImplGpu{}** appears only in the caption, as \(m=1\).

**Caption:** in `final-candidate-opus.tex`.

The natural width is 428 pt, scaled to `\textwidth` (about 347 pt, 81%).
If the scriptsize labels are too small in print, narrow the boxes rather
than shrinking the font.
