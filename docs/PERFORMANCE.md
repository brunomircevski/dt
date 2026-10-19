# Performance

Measurements of `./tree` on its own: what the latest optimizations changed. The current benchmark results (serial, parallel and CUDA on SUSY and HIGGS, and the comparison with scikit-learn, rpart, Weka J48 and YaDT) are in [bench/results/](../bench/results/), described in [bench/README.md](../bench/README.md).

## October 2026: hand-offs without page-locking the arena (CUDA, mordor3 cluster)

The host arena that receives the hand-offs (75 GB for ECBDL'14's 15M sample)
is no longer page-locked. Its registration held the CUDA driver's lock for
1.2–1.9 s, and 4.8–11.5 s on ampere when the node's memory was busy, and the
GPU loop waited for it. Now each hand-off piece is downloaded into one of two
page-locked 128 MB buffers (one per copy stream), and a host function copies
it into the arena on the staging threads while the other stream downloads the
next piece ([CUDA.md](CUDA.md)). Same trees. Medians against e0a9106
([cluster-handoff-staging](../bench/results/cluster-handoff-staging/BENCHMARK.md)):

| Train total (s) | H100 before | H100 after | A100 before | A100 after |
|---|--:|--:|--:|--:|
| ECBDL 15M C4.5 | 52.73 | 51.25 | 74.68 | 65.73 |
| ECBDL 15M CART, no pruning | 18.73 | 17.60 | 31.73 | 23.89 |
| airline C4.5 | 6.19 | 6.08 | 5.95 | 5.65 |
| airline CART, no pruning | 4.19 | 3.31 | 3.79 | 3.65 |
| HIGGS C4.5 | 1.10 | 1.24 | 1.18 | 1.12 |

* The arena is still touched on all threads during the presort: without that,
  the copies faulted its pages in, and airline's levels on the H100 took
  1.8–2.0 s instead of 1.2 s. `MADV_POPULATE_WRITE` instead of the memset holds
  the process's mmap lock and stretched the presort from 0.5 to 4.2 s when the
  node's memory was low.
* 32 MB buffers (four times as many host functions, which run one at a time)
  left the levels 0.2–0.4 s slower than 128 MB ones.
* HIGGS C4.5 on the H100 lost 0.1 s: the pruning's arranged rows now go
  through the staging buffers too.

## October 2026: one copy of the columns (CUDA, serial, parallel)

The columns now live in **slots**, one per feature and a few spare ones, and a
partition moves each column into a free slot instead of ping-ponging between
two full copies ([CUDA.md](CUDA.md), [CPU.md](CPU.md)). The CUDA backend also
dropped its device mirror of the host arena (the hand-offs go through a
512 MB ring) and presorts in the slots. Same trees everywhere (`tests/check.sh`
for `./tree` and `./tree_cpu`, and every run below byte-identical). Means of
3 runs (GPU) or 2 (CPU parallel; serial 1), `train total`, depth 19, no pruning, the
mordor3 nodes; before = 66bd66c.

| CUDA | H100 before | H100 after | A100 before | A100 after |
|---|--:|--:|--:|--:|
| HIGGS CART | 0.39 s, 7.2 GiB | 0.40–0.42 s, 3.4 GiB | 0.49 s, 7.2 GiB | 0.49 s, 3.3 GiB |
| HIGGS C4.5 | 0.36 s, 7.2 GiB | 0.36 s, 3.3 GiB | 0.46 s, 7.1 GiB | 0.44 s, 3.3 GiB |
| airline CART | 1.33 s, 35.0 GiB | 1.38 s, 13.3 GiB | 1.94 s, 35.0 GiB | 1.98 s, 13.3 GiB |
| airline C4.5 | 1.30 s, 35.0 GiB | 1.33 s, 13.3 GiB | 1.87 s, 34.9 GiB | 1.91 s, 13.3 GiB |

* GPU memory: 53% less on HIGGS, 62% less on airline (cuML used 1.9 and
  9.5–10 GiB, [cluster-tree-vs-gpu-libs](../bench/results/cluster-tree-vs-gpu-libs/BENCHMARK.md)).
* Airline is 2–3% slower: the partition is 3% slower (a column waits for the
  one before it to leave its slot, 3–5 ms of 500), and the GPU sometimes
  waits for the ring's copies to the host.
* Traps found on the way: tiles taken column by column instead of node by node
  made the partition 10% slower (the dependency is per segment, so the order
  could stay); a 64-bit field in `PartitionTarget`, indexed by side, pushed it
  to local memory (+5%); a move record read by every thread, or waited for by a
  second thread, +3%; growing CUB's temp buffer in the presort, or a sync per
  feature, put the presort behind the arena's page-locking (C4.5 airline
  presort 0.1 → 0.4 s), so the arena is now registered after the presort.

| CPU (48 threads) | epyc1 before | epyc1 after | dell1 before | dell1 after |
|---|--:|--:|--:|--:|
| HIGGS CART parallel | 1.04 s, 5.7 GiB | 1.06 s, 5.6 GiB | 1.28 s | 1.27 s |
| HIGGS C4.5 parallel | 1.06 s, 5.8 GiB | 1.01 s, 5.7 GiB | 1.50 s | 1.48 s |
| airline CART parallel | 7.05 s, 29.0 GiB | 7.09 s, 27.3 GiB | 7.81 s | 7.53 s |
| airline C4.5 parallel | 8.26 s, 29.0 GiB | 8.12 s, 27.3 GiB | 8.81 s | 8.94 s |
| HIGGS CART serial | 18.1 s, 5.7 GiB | 18.0 s, 3.6 GiB | 25.0 s | 24.8 s |
| HIGGS C4.5 serial | 22.3 s, 5.7 GiB | 20.7 s, 3.6 GiB | 34.1 s | 32.6 s |

* `--serial` uses one spare slot: 37% less memory on HIGGS, and 0–7% faster
  (the winner's column is no longer copied).
* `--parallel` keeps `min(F − 1, threads)` spare slots by default, so that a
  big node moves all its columns at once, one per thread, with the children's
  sweeps fused in: memory drops only where the features outnumber the threads
  (airline: 6%; ECBDL's 631 features on 48 threads need 678 slots instead of
  1262). `--spare-slots N` trades speed for memory (CPU.md).
* The cluster benchmarks were rerun on cd18717 (2026-10-17,
  [bench/results/](../bench/results/README.md)): the same trees everywhere,
  times within a few percent of the previous runs. Full ECBDL'14 on the CPU
  servers needs 237 GiB instead of 375 GiB; its GPU sample grew from 6M to 15M
  rows, which fill 73.5 GiB of the 80 GB GPUs as 6M did before. A same-job
  comparison on the A100 (old and new binary interleaved, 5 runs): pruned
  HIGGS CART 0.50 → 0.51 s, C4.5 1.19 → 1.16 s, airline CART 1.60 → 1.63 s,
  C4.5 5.91 → 5.90 s.

## October 2026: the serial backend runs the parallel algorithm

Until 2026-10-17 `--serial` kept no second copy of the columns, so it
partitioned every node in place, through a byte per row, while `--parallel`
partitioned big nodes (≥ 32768 rows) out of place through a bitmap and swept
the big children during the partition. Now both do the latter (commit
4b1ced0); serial differs only in running on one thread (plain bit operations
instead of atomics). Same trees everywhere. One-thread `./tree` rows of the
benchmarks were rerun with `bench/serial_rerun.py` (median of 3, the same
nodes):

| Serial | epyc1 | dell1 | desktop |
|---|--:|--:|--:|
| airline CART | 148.5 → 76.4 s (1.94×) | 204.7 → 128.1 s (1.60×) | – |
| airline C4.5 | 250.6 → 178.6 s (1.40×) | 338.3 → 264.1 s (1.28×) | – |
| HIGGS CART | 15.3 → 18.2 s (0.84×) | 29.3 → 25.7 s (1.14×) | – |
| HIGGS C4.5 | 30.5 → 37.8 s (0.81×) | 54.4 → 54.0 s (1.01×) | – |
| SUSY CART | 4.68 → 4.98 s (0.94×) | 7.97 → 6.84 s (1.16×) | 4.63 → 4.53 s (1.02×) |
| SUSY C4.5 | 5.83 → 6.77 s (0.86×) | 10.75 → 9.06 s (1.19×) | 5.99 → 5.90 s (1.01×) |

* Airline (115M rows) gains most: the old byte per row (115 MB) misses the
  caches, the bitmap (14 MB) does not.
* On epyc, HIGGS and SUSY got 6–19% slower (HIGGS CART's build 13.55 →
  16.25 s; three runs within 0.05 s). Not investigated; one candidate is the
  non-temporal stores of the out-of-place partition, tuned for many threads.
* Memory: a serial run now holds the second copy too (`8 · F · n` bytes):
  HIGGS 3.7 → 5.7 GiB, airline 20.4 → 29.0 GiB.

## October 2026: profiling airline with Nsight Systems (mordor3 cluster)

Commit `35cf315` (before) against this work (after), on airline, same nodes
and method as below: before and after in the same job, interleaved, 3 runs
each, medians of `train total` (presort + build + prune for C4.5, presort +
build for CART in brackets; on the GPU nodes the total also counts the GPU
setup). Every tree is byte-identical, and
`tests/check.sh` passes for `./tree` and `./tree_cpu`.

| Algorithm | Node, backend | Before | After | Speed-up |
|-----------|---------------|-------:|------:|--------:|
| C4.5 | hopper cuda | 7.29 s (0.54 + 3.29 + 3.22) | 5.19 s (0.09 + 2.88 + 2.02) | 1.40× |
| C4.5 | ampere cuda | 8.48 s (0.70 + 3.34 + 3.89) | 5.85 s (0.11 + 3.30 + 2.12) | 1.45× |
| C4.5 | epyc1 parallel | 12.34 s (0.72 + 10.28 + 1.36) | 11.87 s (0.47 + 10.09 + 1.33) | 1.04× |
| CART | hopper cuda | 1.18 s (0.09 + 0.92) | 1.11 s (0.09 + 0.81) | 1.06× |
| CART | ampere cuda | 1.89 s (0.10 + 1.23) | 1.59 s (0.10 + 1.17) | 1.19× |
| CART | epyc1 parallel | 7.31 s (0.67 + 6.64) | 7.11 s (0.45 + 6.66) | 1.03× |

GPU setup (upload) on ampere went from 0.56 to 0.32 s; on hopper it stays
at about 0.2 s. The time was found with Nsight Systems (`make NVTX=1`, see
below) and `perf` on ampere. What changed:
- **C4.5's distinct values on the device:** the GPU presort used to copy all
  sorted values back for the thresholds; now CUB's `DeviceSelect::Unique`
  keeps each feature's distinct values and only those are copied (presort
  0.54 → 0.09 s on hopper, 0.70 → 0.11 s on ampere). The CPU grower takes
  them from its sorted columns in parallel.
- **Copies on the GPU's NUMA node:** ampere has two sockets and its A100 on
  the first. Staging buffers and copy threads on the other socket ran at
  21 GB/s up and 16 GB/s down, against 25 and 24 on the GPU's own (hopper:
  about 55 GB/s either way). The staging buffers and a pool of copy threads
  are now pinned to the GPU's node (read from sysfs), for the upload, the
  leaves of the rows and the pruning's arrangement.
- **Asynchronous hand-off:** the level loop used to wait for each hand-off's
  copy to the host. With room on the device (airline on both GPUs) the
  columns are gathered into a device arena, copied on a second stream, and
  a host callback starts the CPU tasks; the GPU goes on with the next level.
  Buffer resizes use stream-ordered allocation, so they do not synchronize.
- **The pruning's first arrangement on the GPU:** C4.5 pruning starts by
  sorting the 115M rows by leaf and transposing their features to rows
  (0.4 s on 32-48 threads). When the device can keep a copy of the raw
  values, a radix sort and a gather kernel do it on the GPU and the result
  is copied into the pruning's workspace.
- **Leaf ids renumbered lazily:** the leaf of every row went through two
  renumberings (node store to tree, collapse of useless splits), each a pass
  over 115M ids. Now only a map from old to new ids (one per node) is
  composed, and the ids are read through it.
- **Parallel presort for few features:** with fewer features than threads
  each big column's radix sort is split across threads (digit counts per
  part, prefix over parts, stable parallel scatter): epyc1's presort 0.67 →
  0.45 s. Root class counts are counted in parallel too.

Tried and dropped:
- **Routing the pruning's extra rows on the GPU** (subtree raising counts a
  sibling's rows in the raised branch): the per-call uploads, the scan of
  changed nodes and the calls serialized on one device made the big routes
  slower (0.85 → 2.4 s on hopper).
- **More levels on the GPU** (`--gpu-min-rows` 512 or 1024 instead of 4096
  for C4.5 and 2048 for CART): slower on both GPU nodes (ampere C4.5 build
  3.3 → 5.9 s at 512). A level of many small nodes costs the GPU nearly as
  much as one of big nodes, and the CPU threads finish small subtrees fast.
  The defaults are best on hopper and on ampere, so the benchmarks need no
  per-node settings.
- **Tile sizes** of 4 or 16 entries per thread instead of 8: slower.

The CPU builder's sweep and partition (most of epyc1's 10 s C4.5 build) are
unchanged. The cluster benchmarks were rerun on this commit (`8da0c12`):
[cluster-airlines](../bench/results/cluster-airlines/BENCHMARK.md),
[cluster-higgs](../bench/results/cluster-higgs/BENCHMARK.md) and
[cluster-cpu-scaling](../bench/results/cluster-cpu-scaling/BENCHMARK.md).

### Profiling with Nsight Systems

`make NVTX=1` builds `./tree` with NVTX ranges around the phases (load,
upload, presort, every GPU level, hand-off copies, CPU subtrees, merge,
pruning steps, and every `DT_PROFILE` counter). Then, on a GPU node (not the
login node):

```
nsys profile -t cuda,nvtx,osrt -o airline-c45 ./tree --cuda --c45 train.csv
nsys stats -r nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum airline-c45.nsys-rep
```

CPU sampling (`-s cpu`) needs `perf_event_paranoid` ≤ 2: on mordor3 only
ampere allows it. Without NVTX=1 the ranges compile to nothing.

## October 2026: C4.5 pruning, the GPU partition and the upload (mordor3 cluster)

Commit `93920b9` (before) against the merge of this work (after), on airline
(115M rows, 18 features), measured on hopper (H100 PCIe + 32 Xeon cores,
`--cuda`), ampere (A100 80 GB + 48 EPYC 7402 cores, `--cuda`) and epyc1
(2× EPYC 9255, 48 cores, `--parallel`). Before and after ran in the same job,
interleaved, 3 runs each; times are medians of `train total` (build and prune
in brackets). Every tree is byte-identical before and after, on all three
nodes.

| Algorithm | Node, backend | Before | After | Speed-up |
|-----------|---------------|-------:|------:|--------:|
| C4.5 | hopper cuda | 11.08 s (3.61 + 6.66) | 7.11 s (3.30 + 3.09) | 1.56× |
| C4.5 | ampere cuda | 11.72 s (3.89 + 6.65) | 8.59 s (3.39 + 3.88) | 1.37× |
| C4.5 | epyc1 parallel | 13.78 s (10.16 + 2.90) | 12.33 s (10.26 + 1.37) | 1.12× |
| CART | hopper cuda | 2.17 s (1.72) | 1.21 s (0.92) | 1.79× |
| CART | ampere cuda | 2.88 s (2.28) | 1.86 s (1.24) | 1.55× |
| CART | epyc1 parallel | 7.36 s (6.66) | 7.40 s (6.69) | – |

CART is `--cart --alpha 1e-5`. What changed:
- **The growers record each row's leaf.** C4.5 pruning needed every row's
  leaf and routed all 115M rows through the 19.6M-node tree for it. The
  builders know a node's rows when
  it becomes a leaf, and the ids now follow the tree through its renumbering
  and the collapse of useless splits.
- **C4.5 pruning without blocking joins:** a node is settled by whichever of
  its branches finishes last, so no thread waits. The rows are grouped by
  leaf with a radix sort whose passes have equal digits (two 12-bit passes
  for airline's 4.3M leaves instead of 11 + 11 + 1 bits), and the counts of a
  sibling's rows per leaf go through a small per-thread cache.
- **GPU goes-left flags as a bitmap:** one bit per row (14 MB for airline,
  in L2) instead of a byte (115 MB). The partition kernel runs at about
  1.2 TB/s on the H100; hopper's CART build went from 1.72 to 0.92 s with the
  hand-off below.
- **Staged upload:** with HMM (both GPU nodes) page-locking is slow, so the
  columns go through four page-locked 32 MB staging buffers instead: 6 GB in
  0.2 s instead of 0.35 s on the H100 node, the same 0.5 s on the A100 node.
- **Hand-off arena page-locked as needed:** CART with a fixed α hands only 2
  of airline's 12 GB to the CPU, so its arena is page-locked in 1 GB chunks
  ahead of the hand-offs instead of all at once (0.1-0.3 s). C4.5 keeps the
  single registration: chunks registered during the levels stall the GPU
  loop and made it 0.4 s slower.
- **`--gpu-min-rows` for CART:** rows / 8192, between 1024 and 2048 (airline:
  2048 instead of 4096). C4.5 keeps rows / 4096.
- **`chooseC45`** looks up `log2` in a table and allocates nothing.
- **`DT_PROFILE=1`** prints where C4.5 pruning spends its time.

The parallel CPU build is unchanged (its code is the same; epyc1's C4.5 build
is 1% slower in every binary built after the pruning changes, which we put
down to code layout). The cluster benchmarks were rerun on this commit:
[cluster-airlines](../bench/results/cluster-airlines/BENCHMARK.md),
[cluster-higgs](../bench/results/cluster-higgs/BENCHMARK.md) and
[cluster-cpu-scaling](../bench/results/cluster-cpu-scaling/BENCHMARK.md). The
laptop and desktop benchmarks were not.

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
