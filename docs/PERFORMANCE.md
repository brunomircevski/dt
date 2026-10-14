# Performance

Measurements of `./tree` on its own: what the latest optimizations changed. The current benchmark results (serial, parallel and CUDA on SUSY and HIGGS, and the comparison with scikit-learn, rpart, Weka J48 and YaDT) are in [bench/results/](../bench/results/), described in [bench/README.md](../bench/README.md).

## October 2026: the node store and the GPU hand-off (mordor3 cluster)

Commit `0475249` (node store) and `f495aed` (GPU hand-off), measured on the
cluster's two GPU servers (CUDA) and two CPU servers (parallel), airline
(115M rows), one run per case. Every tree is byte-identical before and after.

- **Node store:** the builders take node ids from a shared store of
  fixed-size chunks. Threads took ids in blocks of 64 from the same chunk,
  so all of them reached the chunk's end together and waited while one
  allocated the next chunk under the lock. On hopper (2 sockets, `schedutil`
  governor) the waiting also let the clocks drop to about 0.9 GHz, and the
  CUDA backend's CPU threads ran half as fast on both sockets as on one.
  Now a thread takes a whole chunk (4096 nodes) and allocates it outside the
  lock, on its own socket.
- **GPU hand-off:** with the CPU subtrees faster, airline is fastest with
  nodes handed to the CPU at 2-4K rows instead of 28K. The automatic
  `--gpu-min-rows` (rows / 4096) is now capped at 4096; datasets up to
  16.8M rows keep their value.

| Airline | Node, backend | Before | Node store | + hand-off |
|---------|---------------|-------:|-----------:|-----------:|
| CART, unpruned | hopper cuda | 14.61 s | 4.98 s | 3.92 s |
| CART, unpruned | ampere cuda | 7.14 s | 5.29 s | 4.66 s |
| C4.5, unpruned | hopper cuda | 7.08 s | 4.82 s | 4.39 s |
| C4.5, unpruned | ampere cuda | 7.13 s | 6.02 s | 5.36 s |
| C4.5 | hopper cuda | 13.76 s | 11.38 s | 11.22 s |
| C4.5 | ampere cuda | 13.63 s | 12.25 s | 11.95 s |
| CART | ampere cuda | 3.34 s | 3.35 s | 2.84 s |
| CART, unpruned | dell parallel | 10.62 s | 10.25 s | – |
| C4.5 | dell parallel | 16.52 s | 16.04 s | – |

One socket is no longer faster anywhere: hopper's unpruned CART takes
6.13 s on one socket, 4.98 s on both. Other settings tried and left at their
defaults (no consistent gain): `--task-rows` 16K-1G, `--feature-parallel-rows`
8K and 128K on the CPU servers, `--gpu-min-rows` 1024 and 16384 on HIGGS.

## October 2026: tuning for many-core servers (mordor3 cluster)

Commit `a8563b3` (before) against `16cc93b` (after), measured on the cluster's
epyc1 node (2× EPYC 9255, 48 cores, `--parallel`) and hopper (H100 PCIe + 32
Xeon cores, `--cuda`). Before and after ran in the same job, interleaved, 3
runs each. Times are medians of `train total`. Every tree is byte-identical
before and after, on both backends.

| Dataset | Algorithm | Node, backend | Before | After | Speed-up |
|---------|-----------|---------------|-------:|------:|--------:|
| HIGGS (10.5M rows) | CART | epyc parallel | 1.95 s | 1.07 s | 1.8× |
| HIGGS | C4.5 | epyc parallel | 2.55 s | 1.86 s | 1.4× |
| airline (115M rows) | CART | epyc parallel | 15.98 s | 7.34 s | 2.2× |
| airline | C4.5 | epyc parallel | 24.67 s | 13.93 s | 1.8× |
| HIGGS | CART | hopper cuda | 1.48 s | 0.54 s | 2.8× |
| HIGGS | C4.5 | hopper cuda | 2.61 s | 1.65 s | 1.6× |
| airline | CART | hopper cuda | 11.64 s | 2.32 s | 5.0× |
| airline | C4.5 | hopper cuda | 26.03 s | 14.53 s | 1.8× |

What changed:
- **CART with a fixed α** does not grow nodes that pruning would cut back
  (docs/ALGORITHMS.md).
- **`--feature-parallel-rows` 4096 → 32768:** fewer thread wake-ups. A 48-core
  build had spent a third of its CPU time in the kernel.
- **Huge pages** for large blocks (`src/core/memory.cpp`).
- **`--gpu-min-rows`:** rows / 4096, at least 1024 (was 512).
- **GPU upload:** a page-locked dataset upload that works on the H100 node.
- **Parallel tree passes:** the copy, compaction and C4.5 bookkeeping run on
  all threads.

The benchmarks on the cluster nodes are in
[cluster-higgs](../bench/results/cluster-higgs/BENCHMARK.md) and
[cluster-airlines](../bench/results/cluster-airlines/BENCHMARK.md). The
laptop and desktop were not re-measured.

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
