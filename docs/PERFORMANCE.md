# Performance

Earlier measurements of `./tree` on its own, moved here from the main README. For the comparison with scikit-learn, rpart, Weka J48 and YaDT, see [bench/README.md](../bench/README.md) and the runs in [bench/results/](../bench/results/).

## October 2026: sweep, partition and pruning

Commit `d1cde64` (before) against `be8b236` (after), one run each, on the
desktop (i7-14700KF, 28 threads, DDR4-3200) and the laptop (i7-13650HX,
20 threads, RTX 5070 Laptop GPU). CART: `--cart -d 30 --alpha 1e-05`, C4.5:
`--c45`. Before and after are `train total` (presort, GPU setup, build,
pruning; loading and evaluation excluded); build, C4.5 pruning and peak
memory (after training) as before → after. Every tree is byte-identical: the dumps of each
dataset and algorithm have the same checksum on every backend and machine,
before and after.

| Dataset | Algorithm | Machine, backend | Before | After | Speed-up | Build (s) | Prune (s) | Peak (GB) |
|---------|-----------|------------------|-------:|------:|--------:|----------:|----------:|----------:|
| SUSY | CART | desktop serial | 9.48 s | 6.24 s | 1.52× | 8.95 → 5.76 | — | 1.1 → 1.1 |
| SUSY | C4.5 | desktop serial | 8.00 s | 6.23 s | 1.28× | 6.54 → 5.16 | 0.92 → 0.52 | 1.7 → 1.4 |
| SUSY | CART | desktop parallel | 1.33 s | 0.970 s | 1.37× | 1.09 → 0.73 | — | 1.8 → 1.7 |
| SUSY | C4.5 | desktop parallel | 1.77 s | 1.29 s | 1.37× | 1.18 → 0.87 | 0.31 → 0.16 | 2.3 → 2.0 |
| HIGGS | CART | desktop parallel | 4.89 s | 3.36 s | 1.46× | 4.05 → 2.59 | — | 6.1 → 5.9 |
| HIGGS | C4.5 | desktop parallel | 8.00 s | 5.69 s | 1.41× | 5.63 → 4.28 | 1.34 → 0.58 | 8.3 → 7.1 |
| SUSY | CART | laptop parallel | 1.30 s | 0.914 s | 1.42× | 1.13 → 0.77 | — | 1.7 → 1.7 |
| SUSY | C4.5 | laptop parallel | 1.39 s | 1.03 s | 1.35× | 0.96 → 0.74 | 0.23 → 0.14 | 2.3 → 2.0 |
| HIGGS | CART | laptop parallel | 4.63 s | 3.32 s | 1.39× | 4.09 → 2.81 | — | 5.4 → 5.9 |
| HIGGS | C4.5 | laptop parallel | 6.58 s | 5.19 s | 1.27× | 4.67 → 4.10 | 1.26 → 0.53 | 7.6 → 7.1 |
| SUSY | CART | laptop cuda | 0.620 s | 0.380 s | 1.63× | 0.52 → 0.31 | — | 1.3 → 1.3 |
| SUSY | C4.5 | laptop cuda | 0.865 s | 0.459 s | 1.88× | 0.51 → 0.23 | 0.23 → 0.14 | 1.9 → 1.6 |
| HIGGS | CART | laptop cuda | 2.00 s | 1.37 s | 1.46× | 1.66 → 1.18 | — | 3.8 → 3.8 |
| HIGGS | C4.5 | laptop cuda | 4.11 s | 1.97 s | 2.09× | 2.22 → 1.16 | 1.45 → 0.54 | 6.1 → 5.0 |

Where it comes from (details in [CPU.md](CPU.md) and [CUDA.md](CUDA.md)):

* CPU sweep: blocks of 16 cuts skipped by a bound on the gain over the
  block (two classes), so most cuts are only counted.
* CPU `--parallel`: big nodes partitioned out of place into a second copy of
  the columns, with the big children swept while their entries are written
  (each column read once per level instead of twice).
* C4.5 pruning: rows kept in leaf order in a row-major copy, routed 16 at a
  time over compact nodes; subtrees whose rows did not change are not pruned
  again after raising.
* Cuda: a single-pass partition that also counts the next level's tile
  histograms; the exact (double precision) pass scores only the few cuts the
  float pass kept per tile; a Gini estimate with a relative error bound;
  C4.5's sorted values copied into memory page-locked during setup.

HIGGS CART on the laptop GPU is limited by the CPU threads finishing the small
subtrees (`--gpu-profile` shows the wait), so part of the GPU's gain does not
show in its total. The second column copy of `--parallel` costs memory at
low thread counts (HIGGS, 4 threads: +1.9 GB); at 20–28 threads it mostly
replaces the presort's per-thread buffers (peak −1.2 to +0.5 GB above).

## Serial, parallel and Cuda backends

Training time (`train total`: presort, GPU setup, build, cross-validation,
pruning; loading and evaluation excluded) on a laptop: i7-13650HX (20 threads),
RTX 5070 Laptop GPU (8 GB, power-limited). The laptop throttles under
sustained load, so expect ±10–15% run-to-run noise (up to 2× on long parallel
runs). Measured with the former `bench/bench.sh`; the new CPU benchmark is in
[bench/README.md](../bench/README.md).

| Dataset | Algorithm | Serial | Parallel | Cuda |
|---------|-----------|-------:|---------:|-----:|
| covertype (581k × 54, 7 classes) | CART, full tree | 1.23 s | 0.21 s | 0.15 s |
| covertype | CART + 10-fold CV | 11.6 s | 2.03 s | 1.44 s |
| covertype | C4.5 | 1.93 s | 0.41 s | 0.26 s |
| SUSY (5M × 18, 2 classes) | CART, full tree (1.3M nodes) | 19.8 s | 2.08 s | 2.02 s |
| SUSY | CART + 10-fold CV | | 51.2 s | 25.1 s |
| SUSY | C4.5 | 16.0 s | 4.42 s | 1.56 s |
| HIGGS (11M × 28, 2 classes) | CART, full tree (3M nodes) | 64.4 s | 7.06 s | 8.56 s |
| HIGGS | CART + 10-fold CV | | 158 s | 85.2 s |
| HIGGS | C4.5 | 74.9 s | 10.7 s | 7.62 s |

## Compared with the previous version

Same machine, same trees (`tests/check.sh`). Parallel and Cuda rows are
interleaved runs of both binaries (best of 3–4), the others single runs.

| Dataset | Algorithm | Backend | Before | After | Speed-up |
|---------|-----------|---------|-------:|------:|--------:|
| covertype | CART | serial | 2.14 s | 1.23 s | 1.7× |
| covertype | CART | parallel | 0.32 s | 0.21 s | 1.5× |
| covertype | CART | cuda | 0.24 s | 0.15 s | 1.6× |
| covertype | CART + 10-fold CV | serial | 22.7 s | 11.6 s | 2.0× |
| covertype | CART + 10-fold CV | parallel | 4.81 s | 2.03 s | 2.4× |
| covertype | CART + 10-fold CV | cuda | 6.62 s | 1.44 s | 4.6× |
| covertype | C4.5 | serial | 3.08 s | 1.93 s | 1.6× |
| covertype | C4.5 | parallel | 1.25 s | 0.41 s | 3.1× |
| covertype | C4.5 | cuda | 0.38 s | 0.26 s | 1.5× |
| SUSY | CART | serial / parallel / cuda | 19.6 / 2.05 / 2.09 s | 19.8 / 2.08 / 2.02 s | 1.0× |
| SUSY | CART + 10-fold CV | parallel | 82.5 s | 51.2 s | 1.6× |
| SUSY | CART + 10-fold CV | cuda | 47.7 s | 25.1 s | 1.9× |
| SUSY | C4.5 | serial / parallel / cuda | 16.3 / 5.07 / 1.65 s | 16.0 / 5.13 / 1.56 s | 1.0× |
| HIGGS | CART + 10-fold CV | parallel | 200 s | 158 s | 1.3× |
| HIGGS | CART + 10-fold CV | cuda | 140 s | 85.2 s | 1.6× |
| HIGGS | CART / C4.5 | all | | | ≈1.0× (within noise) |

Where it comes from: features that are constant in a node are dropped for its
whole subtree (covertype: 44 of 54 features are binary, about half of the
scanned data), cross-validation reuses the presorted columns / GPU buffers
and builds fold row lists without sorting, and the CPU partition no longer
needs a full-size buffer per thread. SUSY and HIGGS have only continuous
features, so a single tree is grown as fast as before. On data-center GPUs
(fast double precision) the Cuda backend scores every cut in one pass instead
of two (`--gpu-sweep`), which this laptop GPU cannot show.
