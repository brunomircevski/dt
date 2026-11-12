# Diagram proposal — Muse Wonder (Round 3)

**Decision: include one figure**, the Hybrid level step, placed in §3.6
(`\ImplHybrid`) as `figs/approach-hybrid.pdf`, referenced from the first
paragraph of that subsection. Single column-spanning (`\textwidth`) figure,
no second visual.

## (a) Why this figure

The Hybrid hand-off is the paper's central mechanism and the only part of
Section 3 whose content is fundamentally *temporal and spatial* rather
than algorithmic: two devices, three buffer stages, and an overlap claim
("the GPU proceeds while the CPU grows handed-off subtrees"). In prose
this takes a full dense paragraph and still leaves readers unsure what
waits for what. A small schematic makes the three load-bearing facts
immediate: (1) assignment is per child node, decided at each level;
(2) the copy path is staged (device ring → pinned buffers → host arena),
not a direct transfer; (3) CPU subtree growth overlaps the *next* GPU
level, it does not interleave with the current one. The shared column
layout of §3.1, by contrast, is adequately served by two sentences
(node = same interval in every column) and does not earn a second panel;
drawing it would spend the figure budget on the least novel content.

## (b) What it must show

1. **One tree level on the GPU (left/top):** a level band containing
   3–4 node boxes of visibly different widths (sample counts), each
   labelled with its size (e.g. `n = 2.1M`, `n = 40k`, `n = 900`).
   A brace or label over the band: "one level, all nodes together —
   sweep → best cut per attribute → stable partition".
2. **The threshold decision:** from the partition, two arrow classes
   leave the level: thick arrows looping back into the next GPU level
   band ("`n ≥ m`: stay on GPU"), and thin arrows leaving downward
   ("`n < m`: hand off"). The threshold `m` (`--gpu-min-rows`) is drawn
   as an explicit decision element (diamond or labelled fork), not
   buried in an arrow label alone.
3. **The staged copy path (middle):** three stacked/sequential boxes —
   "device ring (512 MiB)" → "page-locked buffers (2 × 128 MiB, copy
   streams)" → "host arena (pageable)" — with the restriction
   "only attributes that can still split" noted on the first arrow.
4. **CPU side (right/bottom):** "CPU thread pool" box receiving subtree
   tasks from the arena, labelled "grows whole subtrees
   (same builder as Multicore CPU)". A dashed arrow from the arena
   labelled "host callback on copy stream starts tasks".
5. **The overlap:** a horizontal time axis or a "meanwhile" dashed
   arrow showing the GPU already processing **level d+1** while the CPU
   pool works on the handed-off subtrees of level d. This is the
   figure's single most important element; without it the figure is a
   data-flow diagram of a synchronous scheme.
6. **One tree, not two:** both the GPU path and the CPU box feed a
   single "shared node store → preorder renumbering → one tree" box at
   the bottom/right. This pre-empts the "two trees merged" misreading.
7. **GPU-only as degenerate case:** a one-line annotation, e.g. a small
   note attached to the threshold element: "\ImplGpu{}: `m = 1` — the
   hand-off path is never taken." Do not draw a second configuration.

## (c) What to omit

- Kernel names and the intra-level kernel pipeline (histograms,
  prefix, ballot/shuffle sweep, bitmap marking) — §3.5 prose carries
  this at stage level; drawing it doubles the figure's density and
  dates fastest.
- NUMA staging placement (pinned buffers on the GPU's node) — one
  clause in §3.5; at most a footnote-sized label on the pinned-buffer
  box if space is free, never a separate socket diagram. Drawing two
  sockets invites the NUMA overclaim the text carefully avoids.
- Buffer counts beyond the three stage labels above; no bandwidths,
  no timings, no dataset names, no thread counts.
- The host round trip for split choice (best cuts → host → decision)
  as a separate arrow loop: it happens in *both* GPU configurations
  and would clutter the Hybrid-specific message. The level band's
  label in (b)1 already implies host involvement; if an arrow is
  wanted, a single thin "best cuts / split decision" return arrow on
  the GPU band suffices.
- Pruning, evaluation, and anything after "one tree".

## (d) Caption draft (as used in `draft-muse.tex`)

> One level of \ImplHybrid{}. The GPU evaluates all large nodes of a
> level together and partitions their columns on the device. Children
> below the hand-off threshold are gathered into a device ring, copied
> through page-locked buffers into a host arena, and grown by the CPU
> pool as independent subtrees while the GPU proceeds to the next
> level. Both devices create nodes of the same tree in a shared store,
> which is renumbered in preorder at the end.

(75 words; excluded from the body word count per the author's target.)

## (e) Concrete sketch (TikZ)

Vertical flow, GPU on top, CPU below, time running left→right via the
dashed "next level" loop. Sized for `\textwidth` × ~0.42`\textwidth`.

```tikz
\begin{tikzpicture}[
  font=\small,
  box/.style={draw, rounded corners=2pt, align=center, inner sep=4pt},
  gpu/.style={box, fill=blue!6},
  buf/.style={box, fill=orange!10},
  cpu/.style={box, fill=green!7},
  arr/.style={->, >=stealth, thick},
  thinarr/.style={->, >=stealth},
  dasharr/.style={->, >=stealth, dashed}]

  % --- GPU level d ---
  \node[gpu, minimum width=118mm, minimum height=16mm] (level) at (0,30mm) {};
  \node[anchor=north west, font=\small\bfseries] at (-58mm,37mm)
    {GPU --- level $d$ (all nodes together)};
  \node[gpu] (n1) at (-38mm,28mm) {$n=2.1$M};
  \node[gpu] (n2) at (-13mm,28mm) {$n=40$k};
  \node[gpu] (n3) at (  8mm,28mm) {$n=900$};
  \node[font=\footnotesize] at (38mm,28mm)
    {sweep $\to$ best cut\\ $\to$ stable partition};

  % --- threshold fork ---
  \node[box, diamond, aspect=2.2, inner sep=1pt] (thr) at (-25mm,12mm)
    {$n < m$?};
  \node[font=\footnotesize, align=left] at (18mm,12mm)
    {$m$ = \texttt{--gpu-min-rows}\\ \ImplGpu{}: $m=1$, path never taken};

  % large children: loop to next level (right side, dashed = time)
  \node[gpu, minimum width=34mm] (next) at (38mm,-2mm)
    {GPU --- level $d{+}1$};
  \draw[arr] (level.south east) |- (next.north)
    node[pos=.75, right, font=\footnotesize] {$n \ge m$: stay on GPU};

  % --- copy path ---
  \node[buf] (ring)  at (-38mm,-2mm)  {device ring\\ (512 MiB)};
  \node[buf] (pin)   at (-38mm,-14mm) {page-locked buffers\\ (2 copy streams)};
  \node[buf] (arena) at (-38mm,-26mm) {host arena (pageable)};
  \draw[arr] (thr.south) -- (ring.north)
    node[midway,left,font=\footnotesize] {gather splittable attrs only};
  \draw[arr] (ring) -- (pin);  \draw[arr] (pin) -- (arena);

  % --- CPU pool ---
  \node[cpu, minimum width=40mm, minimum height=12mm] (pool) at (20mm,-22mm)
    {CPU thread pool\\ grows handed-off\\ subtrees};
  \draw[dasharr] (arena.east) -- (pool.west)
    node[midway,above,font=\footnotesize] {callback starts tasks};

  % overlap cue
  \draw[dasharr] (pool.north) -- (next.south)
    node[midway,right,font=\footnotesize,align=left]
    {runs while GPU\\ does level $d{+}1$};

  % --- one tree ---
  \node[box, minimum width=52mm] (tree) at (8mm,-38mm)
    {shared node store $\to$ preorder renumbering $\to$ one tree};
  \draw[thinarr] (next.south) |- (tree.east);
  \draw[thinarr] (pool.south) |- (tree.west);
\end{tikzpicture}
```

Drawing notes for whoever renders it: keep the two "stay / hand off"
arrow classes visually distinct (thick solid vs thin); the dashed
callback and dashed overlap arrows must use the *same* dash style only
if both mean "asynchronous" — otherwise give the overlap cue a grey
background band labelled "concurrent" spanning the CPU box and the
level-d+1 box, which reads better in print than a third arrow style.
Black-and-white safe: distinguish GPU/CPU/buffers by border weight and
position, not colour alone.
