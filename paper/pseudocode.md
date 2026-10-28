# Pseudo code

Notation: n rows, F features, K classes, T = 2048 entries per tile (256
threads × 8), `C_f` the presorted column of feature f, a *segment* = the
contiguous range of one node in one column. Explanations in
[optimizations.md](optimizations.md); real code in `src/build/gpu_grower.cu`
(kernel names given in comments).

## Algorithm 1 — Level-synchronous growth with CPU hand-off

```
Input: X ∈ R^{n×F}, labels y ∈ {0..K-1}^n, depth limit D, hand-off size m
 1  Upload y; upload X through 4 pinned 32 MB staging buffers (GPU's NUMA node),
    on a copy stream, column after column
 2  for f ← 1..F, as soon as column f is uploaded:     ▷ presort, overlapped with step 1
 3      C_f ← StableRadixSort{ (X[i,f], pack(i, y_i)) }    ▷ 8 B/entry, stored in slot f
 4  L ← {root};  counts(root) ← Histogram(y)            ▷ on the GPU, during the presort
 5  while L ≠ ∅:
 6      S ← {(v,f) : v ∈ L, f usable at v}              ▷ segments, cut into tiles of T
 7      if L = {root}: TileHistogram(S)                 ▷ below the root, step 13 did it
 8      SegmentPrefix(S)                                ▷ class counts before each tile
 9      if K = 2: TileBound(S)                          ▷ drop tiles that cannot hold the best cut
10      SweepTiles(S); SegmentBest(S)                   ▷ Alg. 2 (Alg. 3 on slow-FP64 GPUs)
11      copy best cut per (v,f) to host                 ▷ the level's only round trip
        host: pick the split of each v (CART / C4.5 rule);
              children's counts = winning feature's left counts;
              mark leaves (pure, < minChild, depth = D) without touching the GPU
12      MarkGoesLeft: B[row] ← 1 for the rows of the smaller side    ▷ Alg. 5; 1-bit bitmap, fits in L2
13      Partition(S) → children's columns in free slots
                       + next level's tile histograms      ▷ Alg. 4
14      small ← children with < m rows:
              gather → device ring → host (2nd stream);
              a host callback spawns CPU subtree builders; the GPU continues
15      L ← remaining children
16  join CPU tasks; assemble the tree
```

Kernels per level: `segmentConstantKernel`, (`tileHistogramKernel` at the
root), `segmentPrefixKernel`, `tileBoundKernel` (K = 2), `tileExactKernel` or
`tileEstimateKernel` + `segmentMaxKernel` + `tileCandidateKernel` +
`tileExactKernel`, `segmentBestKernel`, `markGoesLeftKernel`,
`partitionKernel`, `copyKernel`.

## Algorithm 2 — Warp-ballot sweep of one tile

256 threads, 8 entries each; no shared-memory scan, no atomics, no thread
walks entries one by one. (`loadTile`, `sweepTile`, `tileExactKernel`)

```
warp w owns entries [256w, 256w+256) of the tile; lane ℓ ∈ [0, 32)
before[k] ← tilePrefix[k] + Σ_{w' < w} warpCount[w'][k]    ▷ shared memory, one __syncthreads
best ← none
for s ← 0..7:
    e ← C[base + 256w + 32s + ℓ]                            ▷ coalesced 8-byte load
    for k ← 0..K-1:
        b ← ballot(e.cls = k)
        left[k] ← before[k] + popc(b & lanesBelow(ℓ))       ▷ class counts left of this cut
        before[k] ← before[k] + popc(b)
    prev ← shfl_up(e.value, 1);  next ← shfl_down(e.value, 1)
    if prev < e.value
       ∧ minChild ≤ pos ≤ n − minChild
       ∧ ¬(same class on both sides ∧ both neighbours are singleton values):  ▷ boundary-point rule
        best ← max(best, (Gain(total, left, pos), pos))
tileBest ← BlockReduce(best)                    ▷ fixed order; ties → lowest position
```

## Algorithm 2b — Tile bound for two classes

(`tileBoundKernel`)

```
for each tile t of segment (v, f):
    a ← class counts before t;  z ← class counts after t
    U_t ← max over the 4 corners c ∈ {a₀,z₀} × {a₁,z₁} of Gain(total, c)   ▷ gain convex in left counts
    ℓ_t ← exact gain of t's first allowed cut                               ▷ a real cut: lower bound
lower(v,f) ← max_t ℓ_t
skip tile t in every sweep if U_t < lower(v,f) − 1e-8
```

## Algorithm 3 — Exact result with mostly single precision

Used only when FP32:FP64 > 4:1 (consumer GPUs); on A100/H100 one exact
double sweep (Alg. 2) runs instead. (`tileEstimateKernel`,
`segmentMaxKernel`, `tileCandidateKernel`, `tileExactKernel`)

```
Pass 1 (float), per tile:
    ĝ(cut) = Σ_k (l_k·n − t_k·n_L)² / (n²·n_L·n_R)       ▷ Gini; differences exact in int64
           (entropy: Σ c·log2(m/c) per side)
    tileMax ← max ĝ;  keep the ≤ 8 cuts with ĝ ≥ cutoff(tileMax) as candidates
segMax ← max over the segment's tiles of tileMax
cutoff ← segMax − (segMax·σ + μ)                         ▷ σ, μ from a proven bound 2B + ε_tie
Pass 2 (double), per tile:
    tileMax < cutoff          → skip
    ≤ 8 candidates            → one warp scores them with the CPU's exact cutGain
    otherwise                 → sweep the tile again in double (Alg. 2)
```

## Algorithm 4 — Single-pass stable partition with fused next-level counts

(`partitionKernel`)

```
t ← atomicAdd(tileCounter, 1)          ▷ dynamic tile ids: predecessors are already running
load T entries (coalesced)
g_j ← (pos_j < nL(v))         if f is v's split feature    ▷ left rows come first there
      B[row_j] ⊕ flip(v)      otherwise                    ▷ flip(v) = 1 if B marks the right rows
(prefix_j, ℓ_t) ← BlockScan(g)
publish state[t] ← (PARTIAL, ℓ_t)      ▷ self-validating 64-bit words, no memory fences
                   (TOTAL if t is the segment's first tile)
thread 0: walk back t−1, t−2, … summing PARTIAL counts until a TOTAL
          publish state[t] ← (TOTAL, before + ℓ_t)
          wait until the column that used the target slot has been read   ▷ column slots
stage in shared memory: left entries first, then right entries
write left run  to dst[before + i]
write right run to dst[nLeft + (start − before) + i]     ▷ both coalesced
while writing: class histogram of each next-level tile (match_any + shared atomics)
               and its cut count (ballot) → the next level skips TileHistogram
```

## Algorithm 5 — Mark goes-left bitmap

(`markGoesLeftKernel`)

```
for each split node v with winning feature f, size n_v and left size nL:
    flip(v) ← (2·nL > n_v)                        ▷ mark the smaller side
    I ← flip(v) ? [nL, n_v) : [0, nL)
    for i ∈ I in segment (v, f), in parallel:
        r ← row(C_f[base_v + i])
        atomicOr(B[r / 32], 1 << (r mod 32))      ▷ ≤ n_v/2 atomics; B cleared every level
```
