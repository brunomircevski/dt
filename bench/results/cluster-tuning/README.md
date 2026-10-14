# Tuning ./tree for the cluster

On 2026-10-14 ./tree's defaults and a few hot spots were tuned for the big
nodes of mordor3, with the two backends that matter there:

- **CPU parallel** on **epyc1**: 2× EPYC 9255 (Zen 5), 48 cores.
- **CUDA** on **hopper**: H100 PCIe, plus 32 Xeon Gold 6444Y cores for the
  small subtrees and the pruning.

The data was HIGGS (10.5M rows × 28 features) and airline (115M rows × 13
features). The protocols are those of the benchmarks: CART `--cart -d 30
--alpha 1e-5`, C4.5 `--c45`.

**Result: the same trees, byte for byte, in less time.**

| Case (same job, interleaved, median of 3) | Before | After | Speed-up |
|---|--:|--:|--:|
| HIGGS CART, epyc parallel | 1.95 s | 1.07 s | 1.8× |
| HIGGS C4.5, epyc parallel | 2.55 s | 1.86 s | 1.4× |
| airline CART, epyc parallel | 15.98 s | 7.34 s | 2.2× |
| airline C4.5, epyc parallel | 24.67 s | 13.93 s | 1.8× |
| HIGGS CART, hopper CUDA | 1.48 s | 0.54 s | 2.8× |
| HIGGS C4.5, hopper CUDA | 2.61 s | 1.65 s | 1.6× |
| airline CART, hopper CUDA | 11.64 s | 2.32 s | 5.0× |
| airline C4.5, hopper CUDA | 26.03 s | 14.53 s | 1.8× |

- **Before:** commit a8563b3; **after:** 16cc93b. Times are `train total`: setup, presort, build and pruning, without reading the CSV.
- Source: [`14-epyc-final-old-vs-new.csv`](14-epyc-final-old-vs-new.csv) and [`14-hopper-final-old-vs-new.csv`](14-hopper-final-old-vs-new.csv).
- Both versions build the same trees: [`14-verify.log`](14-verify.log) compares the dumps of HIGGS and airline, CART and C4.5, parallel and CUDA, and runs the golden tests.
- The benchmarks at the new defaults are in [cluster-higgs](../cluster-higgs/BENCHMARK.md) and [cluster-airlines](../cluster-airlines/BENCHMARK.md).

## How it was measured

[`bench/cluster/tune.sh`](../../cluster/tune.sh) runs a case file (one
./tree command line per case) on one whole node (`--exclusive`):

- **Repetitions:** each case runs three times, and the cases are interleaved, so a slow phase of the node does not hit only one case.
- **Recorded per run:** ./tree's phase times, with the CPU time, minor page faults and context switches from `/usr/bin/time`.
- **Files:** a `NN-<node>-<topic>.cases` file is the input, `.csv` the output, and `.log` holds ./tree's own output when it was kept (GPU profiles, timers).

The binaries in the case files:

| Binary | Code |
|---|---|
| `old/tree`, `old/tree_cpu` | the code before tuning |
| `tree.base`, `tree_cpu.base` | the same, plus timers in the pruning |
| `*.mem`, `*.v2` | the state after a step |
| `./tree`, `./tree_cpu` | the state of that round |

Timers that are not in the repository print `PP` (pruning) or `GS` (GPU setup) lines. perf is not available to users on the cluster (`perf_event_paranoid` = 4), so ./tree's timers did that job.

## What was found, and what changed

1. **Thread wake-ups, `--feature-parallel-rows`: 4096 → 32768.**
   - **The problem:** at 4096, every node of 4096+ rows ran its sweeps and partitions as `parallelFor`s that woke all 47 other threads. A HIGGS CART build on epyc spent 23–26 s of CPU time in the kernel: futex calls, 260k context switches per run. Halving the threads cut the kernel time to 4 s but lost only 10% of speed (01).
   - **The sweep:** 12K–64K rows were all better, 16K–32K best on HIGGS and 32K–64K on airline (02, 03, 11, 12).
   - **Result at 32768:** HIGGS CART 1.78 → 1.35 s, C4.5 2.19 → 1.89 s; kernel time 1–2 s.
   - **Tried and rejected:** making idle workers spin before sleeping changed nothing once the threshold was raised (06).
   - **`--task-rows` 4096** was best already (02, 03).
2. **Huge pages.**
   - Transparent huge pages are on request only (`madvise`) on these nodes, as on most distributions. With glibc's `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`:
     - epyc HIGGS: presort 200 → 120 ms; CART 1.88 → 1.77 s, C4.5 2.47 → 2.24 s (01).
     - hopper CUDA: CART 1.42 → 1.14 s, C4.5 2.42 → 1.92 s (01, 02).
   - `src/core/memory.cpp` now replaces `operator new` so that blocks of 2 MiB or more get them (`madvise(MADV_HUGEPAGE)`), with the same effect as the environment variable (06).
   - Interleaving memory over both sockets (`numactl --interleave=all`) was 20% slower (01).
3. **CART with a fixed alpha stops where pruning would cut** (`SplitRules::isTerminal`).
   - **The rule:** a node with fewer than `alpha · n` training errors becomes a leaf, since cost-complexity pruning would collapse any subtree under it (docs/ALGORITHMS.md).
   - **Work saved:** HIGGS grows 116 thousand nodes instead of 2.2 million, airline 0.1 instead of 21.6 million.
   - **Result:** epyc HIGGS CART 1.37 → 1.10 s, hopper CUDA 1.13 → 0.51 s (07).
4. **GPU hand-off size, `--gpu-min-rows`: 512 → training rows / 4096, at least 1024.**
   - **Small values lose:** below 1024, the GPU spends its time on levels of tiny nodes (01).
   - **Big values win on big data:** a GPU level costs about as much for thousands of mid-size nodes as for a few big ones (10, 12, 13). Handing those nodes to the CPU earlier helps:
     - airline C4.5: 21.8 s at 1024, 17.2–17.6 s at 16K–64K;
     - HIGGS: flat from 1K to 4K, worse at 16K;
     - CART: best at 1K–2K. At 16K, HIGGS's CART is 50% slower and airline's 5%.
   - The rule gives 2,563 rows for HIGGS and 27,970 for airline.
5. **Upload: plain page-locking.**
   - The upload page-locked the dataset with `cudaHostRegisterReadOnly`, which fails on hopper ("operation not supported": the H100 node uses HMM). The copy then ran pageable at about 10 GB/s.
   - A plain registration takes 18 ms per 1.2 GB and copies at 55 GB/s (10, `10-pin.cu`).
   - GPU setup: HIGGS 170 → 100 ms, airline 716 → 360 ms.
6. **Serial tree passes made parallel.** C4.5's pruning on hopper was mostly serial: with 32 threads it took 0.91 s, with 16 threads 1.19 s, so about 0.6 s did not scale (02, 05). Fixed:
   - **Subtree raising's count of a big sibling's rows:** counted by all threads, and summed in the serial order, so the estimate is bit-identical (05).
   - **Passes over the whole tree:** they now run on all threads (`Tree::bottomUp`, parallel `Tree::assignPreorder`). The serial times on airline's 19.6M-node tree on hopper (13):
     - C4.5's collapse with its compaction: 0.8 s;
     - the final compaction: 0.4 s;
     - the first arrangement's preorder list and counts: 0.4 + 0.3 s;
     - `NodeStore::toTree`, the same copy at the end of every build: not timed on its own.
7. **The host CPU matters for CUDA.**
   - One thread on hopper's Xeon 6444Y is half as fast as one on epyc's Zen 5: a serial CART build takes 18.2 s against 9.5 s (05).
   - What the CPU does in a CUDA run therefore weighs more on hopper: C4.5's pruning (0.77 s of HIGGS's 1.65 s, 7.4 s of airline's 14.5 s) and the small subtrees.
   - The thread pool uses all 32 cores; 16 were 30–50% slower (02, 05).
   - Two-pass sweeps make no difference on the H100 (01), so the one-pass default stays.

Not changed, and why:
- **Small datasets on small machines** (SUSY 500k on the desktop) were faster with `--feature-parallel-rows 4096` (docs/CPU.md); the flag is still there.
- **The laptop and desktop** were not re-measured: the laptop was throttled that day.
- **Hopper's CPU timings drift** during the day by up to 40% with the same binary and no other process on the node. Every before/after comparison above is therefore made inside one job, with the runs interleaved.

## All rounds

Medians of every round (`NN`: the round's number above; the `.csv` files have every run).

#### 01-epyc-threads-hugepages-numa.csv

`epyc1, 48 CPUs, job 156, 2026-10-14 14:44:34, load 0.00 0.00 0.00`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-t48 | 3 | 1.88 s (1.87 s–1.90 s) | 208 ms | 1.67 s | 12 ms | 25.6 s |
| c45-t48 | 3 | 2.47 s (2.44 s–2.48 s) | 237 ms | 1.86 s | 367 ms | 20.5 s |
| cart-t48-thp | 3 | 1.77 s (1.75 s–1.77 s) | 118 ms | 1.64 s | 9 ms | 21.9 s |
| c45-t48-thp | 3 | 2.24 s (2.18 s–2.29 s) | 143 ms | 1.79 s | 301 ms | 15.5 s |
| cart-t48-il | 3 | 2.30 s (2.18 s–2.32 s) | 435 ms | 1.86 s | 13 ms | 39.6 s |
| c45-t48-il | 3 | 2.80 s (2.75 s–2.86 s) | 555 ms | 1.86 s | 388 ms | 27.8 s |
| cart-t24-n0 | 3 | 2.09 s (2.06 s–2.09 s) | 219 ms | 1.86 s | 12 ms | 3.9 s |
| c45-t24-n0 | 3 | 3.01 s (2.97 s–3.02 s) | 244 ms | 2.36 s | 398 ms | 4.0 s |
| cart-t36 | 3 | 2.08 s (2.08 s–2.14 s) | 208 ms | 1.86 s | 12 ms | 14.8 s |
| c45-t36 | 3 | 2.78 s (2.74 s–2.79 s) | 237 ms | 2.14 s | 406 ms | 13.6 s |

#### 01-hopper-gpu-min-rows.csv

`hopper, 32 CPUs, job 157, 2026-10-14 14:44:34, load 0.57 0.64 0.35`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-base | 3 | 1.42 s (1.37 s–1.46 s) | 20 ms | 1.21 s | 19 ms | 7.1 s |
| cart-min128 | 3 | 2.01 s (1.76 s–2.23 s) | 20 ms | 1.78 s | 18 ms | 8.0 s |
| cart-min256 | 3 | 1.55 s (1.49 s–1.57 s) | 20 ms | 1.34 s | 18 ms | 7.3 s |
| cart-min1024 | 3 | 1.38 s (1.37 s–1.41 s) | 20 ms | 1.17 s | 20 ms | 6.8 s |
| cart-min2048 | 3 | 1.37 s (1.35 s–1.39 s) | 20 ms | 1.15 s | 25 ms | 6.5 s |
| cart-min4096 | 3 | 1.43 s (1.41 s–1.43 s) | 20 ms | 1.20 s | 23 ms | 6.5 s |
| cart-min16384 | 3 | 2.15 s (2.11 s–2.23 s) | 20 ms | 1.94 s | 23 ms | 17.0 s |
| cart-thp | 3 | 1.14 s (1.13 s–1.17 s) | 20 ms | 948 ms | 13 ms | 3.4 s |
| cart-twopass | 3 | 1.46 s (1.42 s–1.46 s) | 20 ms | 1.25 s | 20 ms | 7.1 s |
| c45-base | 3 | 2.42 s (2.36 s–2.43 s) | 167 ms | 969 ms | 1.11 s | 10.2 s |
| c45-min128 | 3 | 2.82 s (2.80 s–2.82 s) | 163 ms | 1.35 s | 1.12 s | 10.7 s |
| c45-min256 | 3 | 2.48 s (2.45 s–2.50 s) | 166 ms | 1.06 s | 1.09 s | 10.3 s |
| c45-min1024 | 3 | 2.38 s (2.35 s–2.41 s) | 160 ms | 944 ms | 1.12 s | 10.9 s |
| c45-min2048 | 3 | 2.32 s (2.30 s–2.34 s) | 166 ms | 891 ms | 1.09 s | 9.9 s |
| c45-min4096 | 3 | 2.35 s (2.33 s–2.40 s) | 161 ms | 944 ms | 1.09 s | 9.9 s |
| c45-min16384 | 3 | 3.15 s (3.06 s–3.28 s) | 162 ms | 1.72 s | 1.10 s | 21.5 s |
| c45-thp | 3 | 1.92 s (1.89 s–1.94 s) | 125 ms | 738 ms | 891 ms | 3.8 s |
| c45-twopass | 3 | 2.37 s (2.37 s–2.44 s) | 162 ms | 967 ms | 1.07 s | 10.0 s |

#### 02-epyc-feature-parallel-task-rows.csv

`epyc1, 48 CPUs, job 158, 2026-10-14 14:46:30, load 15.17 7.31 2.76`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-base | 3 | 1.78 s (1.76 s–1.79 s) | 116 ms | 1.65 s | 9 ms | 22.0 s |
| cart-fp16384 | 3 | 1.33 s (1.31 s–1.36 s) | 112 ms | 1.21 s | 9 ms | 1.5 s |
| cart-fp65536 | 3 | 1.37 s (1.37 s–1.40 s) | 108 ms | 1.26 s | 9 ms | 1.0 s |
| cart-fp262144 | 3 | 1.47 s (1.44 s–1.49 s) | 107 ms | 1.36 s | 9 ms | 0.9 s |
| cart-fp1048576 | 3 | 1.76 s (1.70 s–1.76 s) | 112 ms | 1.63 s | 9 ms | 1.0 s |
| cart-tr1024 | 3 | 1.97 s (1.94 s–1.98 s) | 109 ms | 1.85 s | 9 ms | 29.1 s |
| cart-tr16384 | 3 | 1.80 s (1.69 s–1.86 s) | 111 ms | 1.69 s | 9 ms | 26.3 s |
| cart-tr65536 | 3 | 1.79 s (1.77 s–1.96 s) | 121 ms | 1.67 s | 9 ms | 20.0 s |
| c45-base | 3 | 2.19 s (2.17 s–2.33 s) | 126 ms | 1.75 s | 310 ms | 15.2 s |
| c45-fp16384 | 3 | 1.89 s (1.88 s–1.89 s) | 129 ms | 1.46 s | 302 ms | 1.6 s |
| c45-fp65536 | 3 | 1.96 s (1.92 s–1.98 s) | 125 ms | 1.54 s | 301 ms | 1.0 s |
| c45-fp262144 | 3 | 2.15 s (2.14 s–2.18 s) | 123 ms | 1.72 s | 308 ms | 1.1 s |
| c45-fp1048576 | 3 | 3.23 s (3.22 s–3.27 s) | 134 ms | 2.80 s | 297 ms | 1.4 s |
| c45-tr1024 | 3 | 2.20 s (2.16 s–2.42 s) | 120 ms | 1.78 s | 298 ms | 14.9 s |
| c45-tr16384 | 3 | 2.21 s (2.19 s–2.25 s) | 122 ms | 1.78 s | 309 ms | 13.7 s |
| c45-tr65536 | 3 | 2.36 s (2.31 s–2.38 s) | 129 ms | 1.92 s | 308 ms | 13.9 s |

#### 02-hopper-gpu-profile.csv

`hopper, 32 CPUs, job 160, 2026-10-14 14:49:36, load 16.81 9.84 4.34`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-prof | 1 | 1.12 s (1.12 s–1.12 s) | 20 ms | 956 ms | 14 ms | 3.4 s |
| c45-prof | 1 | 1.91 s (1.91 s–1.91 s) | 124 ms | 745 ms | 861 ms | 3.7 s |

#### 02-hopper-hugepages-min-rows-threads.csv

`hopper, 32 CPUs, job 159, 2026-10-14 14:47:34, load 8.24 5.29 2.27`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-thp | 3 | 1.16 s (1.09 s–1.16 s) | 20 ms | 948 ms | 14 ms | 3.8 s |
| cart-thp-min1024 | 3 | 1.12 s (1.12 s–1.14 s) | 20 ms | 923 ms | 15 ms | 3.2 s |
| cart-thp-min2048 | 3 | 1.13 s (1.12 s–1.13 s) | 20 ms | 927 ms | 15 ms | 3.2 s |
| cart-thp-min4096 | 3 | 1.15 s (1.13 s–1.16 s) | 20 ms | 958 ms | 15 ms | 3.3 s |
| cart-thp-min8192 | 3 | 1.57 s (1.52 s–1.60 s) | 20 ms | 1.37 s | 15 ms | 7.7 s |
| cart-thp-t16 | 3 | 1.51 s (1.50 s–1.57 s) | 20 ms | 1.32 s | 16 ms | 2.9 s |
| cart-thp-par32 | 3 | 2.81 s (2.77 s–2.84 s) | 271 ms | 2.54 s | 14 ms | 13.7 s |
| c45-thp | 3 | 1.93 s (1.93 s–1.93 s) | 121 ms | 740 ms | 908 ms | 3.7 s |
| c45-thp-min1024 | 3 | 1.92 s (1.85 s–1.92 s) | 122 ms | 737 ms | 878 ms | 3.7 s |
| c45-thp-min2048 | 3 | 1.90 s (1.89 s–1.91 s) | 122 ms | 729 ms | 881 ms | 3.7 s |
| c45-thp-min4096 | 3 | 1.89 s (1.83 s–1.94 s) | 123 ms | 746 ms | 866 ms | 3.6 s |
| c45-thp-min8192 | 3 | 2.25 s (2.20 s–2.29 s) | 124 ms | 1.08 s | 890 ms | 7.3 s |
| c45-thp-t16 | 3 | 2.50 s (2.47 s–2.60 s) | 133 ms | 1.04 s | 1.19 s | 3.5 s |
| c45-thp-par32 | 3 | 4.07 s (4.00 s–4.08 s) | 275 ms | 2.94 s | 797 ms | 9.0 s |

#### 03-epyc-feature-parallel-fine.csv

`epyc1, 48 CPUs, job 161, 2026-10-14 14:48:49, load 20.92 14.93 6.38`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-fp8192 | 3 | 1.39 s (1.36 s–1.43 s) | 108 ms | 1.27 s | 9 ms | 4.7 s |
| cart-fp12288 | 3 | 1.35 s (1.32 s–1.37 s) | 111 ms | 1.23 s | 9 ms | 2.3 s |
| cart-fp16384 | 3 | 1.34 s (1.32 s–1.36 s) | 109 ms | 1.21 s | 9 ms | 1.5 s |
| cart-fp24576 | 3 | 1.36 s (1.32 s–1.36 s) | 119 ms | 1.23 s | 9 ms | 1.1 s |
| cart-fp32768 | 3 | 1.36 s (1.35 s–1.41 s) | 125 ms | 1.23 s | 9 ms | 1.1 s |
| cart-fp16384-tr8192 | 3 | 1.35 s (1.29 s–1.39 s) | 112 ms | 1.23 s | 9 ms | 1.4 s |
| cart-fp16384-tr2048 | 3 | 1.34 s (1.32 s–1.44 s) | 112 ms | 1.22 s | 9 ms | 1.6 s |
| cart-fp16384-nothp | 3 | 1.51 s (1.47 s–1.52 s) | 206 ms | 1.30 s | 12 ms | 5.3 s |
| c45-fp8192 | 3 | 1.98 s (1.97 s–2.09 s) | 130 ms | 1.54 s | 310 ms | 4.5 s |
| c45-fp12288 | 3 | 1.89 s (1.89 s–1.94 s) | 123 ms | 1.47 s | 304 ms | 2.7 s |
| c45-fp16384 | 3 | 1.94 s (1.91 s–1.94 s) | 127 ms | 1.51 s | 304 ms | 1.6 s |
| c45-fp24576 | 3 | 1.94 s (1.90 s–1.94 s) | 143 ms | 1.49 s | 303 ms | 1.2 s |
| c45-fp32768 | 3 | 1.88 s (1.88 s–1.89 s) | 127 ms | 1.46 s | 302 ms | 1.1 s |
| c45-fp16384-tr8192 | 3 | 1.90 s (1.89 s–1.90 s) | 130 ms | 1.46 s | 305 ms | 1.6 s |
| c45-fp16384-tr2048 | 3 | 1.87 s (1.86 s–1.89 s) | 136 ms | 1.43 s | 297 ms | 1.6 s |
| c45-fp16384-nothp | 3 | 2.13 s (2.13 s–2.20 s) | 242 ms | 1.52 s | 372 ms | 6.7 s |

#### 04-epyc-prune-chunks.csv

`epyc1, 48 CPUs, job 164, 2026-10-14 14:51:32, load 11.09 17.23 8.96`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| c45-pr50000-rc64 | 2 | 1.96 s (1.95 s–1.97 s) | 136 ms | 1.44 s | 379 ms | 1.6 s |
| c45-pr50000-rc256 | 2 | 1.92 s (1.91 s–1.94 s) | 133 ms | 1.41 s | 385 ms | 1.3 s |
| c45-pr12500-rc64 | 2 | 1.93 s (1.92 s–1.94 s) | 120 ms | 1.42 s | 394 ms | 1.6 s |
| c45-pr12500-rc256 | 2 | 1.94 s (1.91 s–1.97 s) | 139 ms | 1.42 s | 384 ms | 1.7 s |

#### 04-hopper-prune-chunks.csv

`hopper, 32 CPUs, job 163, 2026-10-14 14:51:32, load 2.56 6.72 3.84`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| c45-pr50000-rc64 | 2 | 2.05 s (2.01 s–2.09 s) | 123 ms | 722 ms | 1.04 s | 3.9 s |
| c45-pr50000-rc256 | 2 | 2.05 s (2.05 s–2.06 s) | 120 ms | 722 ms | 1.05 s | 3.7 s |
| c45-pr12500-rc64 | 2 | 2.18 s (2.16 s–2.19 s) | 120 ms | 722 ms | 1.17 s | 3.7 s |
| c45-pr12500-rc256 | 2 | 2.07 s (2.06 s–2.09 s) | 128 ms | 716 ms | 1.06 s | 3.7 s |
| c45-pr200000-rc64 | 2 | 2.17 s (2.13 s–2.22 s) | 121 ms | 718 ms | 1.17 s | 3.8 s |
| c45-pr200000-rc256 | 2 | 2.13 s (2.11 s–2.15 s) | 123 ms | 724 ms | 1.12 s | 4.0 s |

#### 05-hopper-prune-breakdown-threads.csv

`hopper, 32 CPUs, job 174, 2026-10-14 15:07:59, load 0.70 5.31 5.28`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| c45-t32 | 2 | 1.83 s (1.66 s–2.00 s) | 113 ms | 679 ms | 892 ms | 3.7 s |
| c45-t16 | 2 | 2.48 s (2.48 s–2.48 s) | 77 ms | 1.06 s | 1.18 s | 3.4 s |
| c45-t8 | 2 | 4.20 s (4.11 s–4.28 s) | 84 ms | 1.72 s | 2.20 s | 3.0 s |

#### 06-epyc-operator-new-spinning.csv

`epyc1, 48 CPUs, job 169, 2026-10-14 14:57:10, load 0.22 6.86 6.99`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-base-glibcthp | 3 | 1.36 s (1.34 s–1.36 s) | 113 ms | 1.23 s | 9 ms | 1.6 s |
| cart-new | 3 | 1.34 s (1.34 s–1.36 s) | 114 ms | 1.22 s | 9 ms | 1.6 s |
| cart-new-glibcthp | 3 | 1.36 s (1.33 s–1.37 s) | 109 ms | 1.24 s | 8 ms | 1.5 s |
| cart-new-spin2000 | 3 | 1.34 s (1.33 s–1.35 s) | 110 ms | 1.22 s | 9 ms | 1.5 s |
| cart-new-spin20000 | 3 | 1.34 s (1.33 s–1.34 s) | 109 ms | 1.21 s | 9 ms | 1.5 s |
| cart-new-fp4096-spin20000 | 3 | 1.76 s (1.74 s–1.86 s) | 110 ms | 1.64 s | 9 ms | 22.6 s |
| cart-new-fp8192-spin20000 | 3 | 1.41 s (1.39 s–1.42 s) | 109 ms | 1.30 s | 9 ms | 5.4 s |
| c45-base-glibcthp | 3 | 1.92 s (1.89 s–1.98 s) | 126 ms | 1.40 s | 385 ms | 1.5 s |
| c45-new | 3 | 1.94 s (1.93 s–1.95 s) | 124 ms | 1.41 s | 394 ms | 1.6 s |
| c45-new-glibcthp | 3 | 1.92 s (1.90 s–1.95 s) | 141 ms | 1.39 s | 374 ms | 1.6 s |
| c45-new-spin2000 | 3 | 1.91 s (1.91 s–1.93 s) | 124 ms | 1.40 s | 386 ms | 1.5 s |
| c45-new-spin20000 | 3 | 1.90 s (1.89 s–1.92 s) | 126 ms | 1.39 s | 395 ms | 1.7 s |
| c45-new-fp4096-spin20000 | 3 | 2.22 s (2.21 s–2.25 s) | 122 ms | 1.69 s | 401 ms | 15.8 s |
| c45-new-fp8192-spin20000 | 3 | 1.98 s (1.97 s–1.99 s) | 130 ms | 1.46 s | 390 ms | 4.2 s |

#### 06-hopper-operator-new-spinning.csv

`hopper, 32 CPUs, job 168, 2026-10-14 14:57:10, load 1.04 3.60 3.43`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-base-glibcthp | 3 | 1.12 s (948 ms–1.13 s) | 20 ms | 924 ms | 15 ms | 3.2 s |
| cart-new | 3 | 1.10 s (1.09 s–1.12 s) | 20 ms | 909 ms | 17 ms | 3.2 s |
| cart-new-spin2000 | 3 | 1.11 s (1.11 s–1.13 s) | 20 ms | 909 ms | 15 ms | 3.4 s |
| cart-new-spin20000 | 3 | 1.12 s (1.11 s–1.14 s) | 20 ms | 911 ms | 15 ms | 3.3 s |
| cart-new-par-fp16384 | 3 | 2.35 s (2.32 s–2.41 s) | 261 ms | 2.08 s | 16 ms | 3.1 s |
| cart-new-par-fp16384-spin20000 | 3 | 2.36 s (2.34 s–2.47 s) | 255 ms | 2.08 s | 15 ms | 3.1 s |
| c45-base-glibcthp | 3 | 2.09 s (1.99 s–2.12 s) | 120 ms | 723 ms | 1.09 s | 4.0 s |
| c45-new | 3 | 2.11 s (2.03 s–2.21 s) | 113 ms | 721 ms | 1.11 s | 3.8 s |
| c45-new-spin2000 | 3 | 2.14 s (2.10 s–2.20 s) | 114 ms | 719 ms | 1.15 s | 3.9 s |
| c45-new-spin20000 | 3 | 1.99 s (1.96 s–2.00 s) | 113 ms | 712 ms | 995 ms | 3.8 s |
| c45-new-par-fp16384 | 3 | 3.91 s (3.85 s–4.03 s) | 284 ms | 2.56 s | 1.09 s | 3.8 s |
| c45-new-par-fp16384-spin20000 | 3 | 3.90 s (3.82 s–3.99 s) | 275 ms | 2.50 s | 1.13 s | 3.7 s |

#### 07-epyc-cart-alpha-rule.csv

`epyc1, 48 CPUs, job 172, 2026-10-14 15:02:20, load 0.69 7.08 7.75`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-par-mem | 3 | 1.36 s (1.34 s–1.37 s) | 120 ms | 1.23 s | 9 ms | 1.6 s |
| cart-par-alpha-fp4096 | 3 | 1.63 s (1.61 s–1.76 s) | 121 ms | 1.49 s | 1 ms | 26.3 s |
| cart-par-alpha-fp16384 | 3 | 1.10 s (1.06 s–1.11 s) | 118 ms | 978 ms | 0 ms | 1.6 s |
| cart-par-alpha-fp65536 | 3 | 1.10 s (1.09 s–1.12 s) | 110 ms | 988 ms | 1 ms | 0.9 s |
| c45-par-alpha-fp16384 | 3 | 1.93 s (1.90 s–1.95 s) | 124 ms | 1.41 s | 398 ms | 1.5 s |

#### 07-hopper-cart-alpha-rule.csv

`hopper, 32 CPUs, job 171, 2026-10-14 15:02:20, load 1.41 4.59 4.29`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-par-base-fp4096 | 3 | 3.14 s (2.23 s–3.24 s) | 464 ms | 2.66 s | 20 ms | 20.2 s |
| cart-par-base-fp16384 | 3 | 2.04 s (2.03 s–2.77 s) | 277 ms | 1.75 s | 19 ms | 5.2 s |
| cart-par-mem-fp4096 | 3 | 2.08 s (2.04 s–2.50 s) | 247 ms | 1.83 s | 14 ms | 5.8 s |
| cart-par-mem-fp16384 | 3 | 1.88 s (1.86 s–1.91 s) | 237 ms | 1.63 s | 14 ms | 2.1 s |
| c45-par-base-fp4096 | 3 | 3.56 s (3.55 s–3.71 s) | 284 ms | 2.33 s | 964 ms | 8.7 s |
| c45-par-base-fp16384 | 3 | 3.50 s (3.43 s–3.58 s) | 288 ms | 2.18 s | 1.02 s | 6.5 s |
| c45-par-mem-fp4096 | 3 | 4.20 s (3.28 s–4.39 s) | 242 ms | 2.92 s | 1.04 s | 9.5 s |
| c45-par-mem-fp16384 | 3 | 3.88 s (3.17 s–3.92 s) | 281 ms | 2.58 s | 997 ms | 3.6 s |
| cart-cuda-mem | 3 | 1.13 s (1.12 s–1.13 s) | 20 ms | 910 ms | 15 ms | 3.6 s |
| cart-cuda-alpha | 3 | 517 ms (513 ms–523 ms) | 20 ms | 324 ms | 1 ms | 2.8 s |
| cart-cuda-alpha-min512 | 3 | 535 ms (525 ms–540 ms) | 20 ms | 341 ms | 1 ms | 2.9 s |
| cart-cuda-alpha-min1024 | 3 | 505 ms (505 ms–509 ms) | 20 ms | 312 ms | 1 ms | 2.9 s |
| cart-cuda-alpha-min4096 | 3 | 554 ms (548 ms–558 ms) | 20 ms | 359 ms | 1 ms | 2.8 s |
| cart-cuda-alpha-min8192 | 3 | 1.10 s (1.10 s–1.15 s) | 20 ms | 911 ms | 1 ms | 12.2 s |
| cart-par-alpha-fp16384 | 3 | 1.58 s (1.53 s–1.76 s) | 264 ms | 1.31 s | 1 ms | 3.0 s |

#### 08-epyc-old-vs-new.csv

`epyc1, 48 CPUs, job 178, 2026-10-14 15:12:31, load 0.00 1.28 4.46`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-old | 5 | 1.91 s (1.88 s–2.07 s) | 209 ms | 1.69 s | 11 ms | 26.0 s |
| cart-new | 5 | 1.05 s (1.03 s–1.08 s) | 114 ms | 935 ms | 1 ms | 1.7 s |
| c45-old | 5 | 2.50 s (2.45 s–2.60 s) | 234 ms | 1.80 s | 451 ms | 19.9 s |
| c45-new | 5 | 1.93 s (1.89 s–1.95 s) | 122 ms | 1.46 s | 337 ms | 1.8 s |

#### 08-hopper-old-vs-new.csv

`hopper, 32 CPUs, job 177, 2026-10-14 15:16:42, load 7.99 3.63 4.08`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| cart-cuda-old | 5 | 1.47 s (1.42 s–1.52 s) | 20 ms | 1.25 s | 19 ms | 7.1 s |
| cart-cuda-new | 5 | 511 ms (502 ms–517 ms) | 20 ms | 317 ms | 1 ms | 2.8 s |
| cart-par-old | 5 | 3.22 s (3.20 s–3.34 s) | 461 ms | 2.75 s | 23 ms | 20.4 s |
| cart-par-new | 5 | 1.58 s (1.56 s–1.60 s) | 260 ms | 1.32 s | 1 ms | 2.9 s |
| c45-cuda-old | 5 | 2.64 s (2.62 s–2.70 s) | 166 ms | 990 ms | 1.33 s | 10.6 s |
| c45-cuda-new | 5 | 2.01 s (1.85 s–2.06 s) | 112 ms | 721 ms | 1.01 s | 4.0 s |
| c45-par-old | 5 | 4.99 s (4.85 s–5.16 s) | 511 ms | 3.24 s | 1.26 s | 21.6 s |
| c45-par-new | 5 | 3.79 s (3.71 s–3.84 s) | 283 ms | 2.60 s | 910 ms | 3.9 s |

#### 09-epyc-airline-first.csv

`epyc1, 48 CPUs, job 183, 2026-10-14 15:20:02, load 0.02 1.61 3.86`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| air-cart-par-new | 1 | 7.61 s (7.61 s–7.61 s) | 725 ms | 6.88 s | 0 ms | 7.2 s |
| air-c45-par-new | 1 | 15.73 s (15.73 s–15.73 s) | 754 ms | 11.24 s | 3.74 s | 10.3 s |
| air-cart-par-old | 1 | 15.21 s (15.21 s–15.21 s) | 1.86 s | 13.23 s | 114 ms | 160.5 s |
| air-c45-par-old | 1 | 24.67 s (24.67 s–24.67 s) | 2.09 s | 17.09 s | 5.49 s | 227.2 s |

#### 09-hopper-airline-first.csv

`hopper, 32 CPUs, job 182, 2026-10-14 15:20:01, load 5.50 7.31 5.67`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| air-cart-cuda-new | 1 | 2.45 s (2.45 s–2.45 s) | 87 ms | 1.65 s | 1 ms | 13.1 s |
| air-c45-cuda-new | 1 | 23.36 s (23.36 s–23.36 s) | 514 ms | 12.52 s | 9.67 s | 19.3 s |
| air-cart-cuda-old | 1 | 11.01 s (11.01 s–11.01 s) | 87 ms | 10.01 s | 230 ms | 53.7 s |
| air-c45-cuda-old | 1 | 31.22 s (31.22 s–31.22 s) | 842 ms | 12.68 s | 17.02 s | 57.2 s |

#### 10-hopper-airline-gpu-profile.csv

`hopper, 32 CPUs, job 185, 2026-10-14 15:23:46, load 1.87 5.79 5.56`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| air-c45-prof | 1 | 20.79 s (20.79 s–20.79 s) | 517 ms | 11.45 s | 8.19 s | 17.4 s |
| higgs-cart-gs | 1 | 502 ms (502 ms–502 ms) | 20 ms | 312 ms | 1 ms | 2.9 s |
| higgs-c45-gs | 1 | 2.05 s (2.05 s–2.05 s) | 113 ms | 734 ms | 1.04 s | 3.7 s |

#### 10-hopper-airline-min-rows.csv

`hopper, 32 CPUs, job 186, 2026-10-14 15:24:17, load 9.15 7.19 6.03`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| air-c45-min256 | 1 | 22.33 s (22.33 s–22.33 s) | 517 ms | 12.07 s | 9.09 s | 19.8 s |
| air-c45-min512 | 1 | 19.13 s (19.13 s–19.13 s) | 513 ms | 9.42 s | 8.56 s | 15.3 s |
| air-c45-min2048 | 1 | 18.89 s (18.89 s–18.89 s) | 513 ms | 9.21 s | 8.53 s | 15.9 s |
| air-c45-min4096 | 1 | 21.50 s (21.50 s–21.50 s) | 514 ms | 11.22 s | 9.12 s | 17.3 s |
| air-c45-min16384 | 1 | 18.06 s (18.06 s–18.06 s) | 518 ms | 7.72 s | 9.18 s | 14.9 s |

#### 11-epyc-airline-feature-parallel.csv

`epyc1, 48 CPUs, job 187, 2026-10-14 15:23:50, load 2.31 5.85 5.51`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| air-cart-fp8192 | 2 | 8.33 s (8.23 s–8.44 s) | 662 ms | 7.67 s | 0 ms | 37.9 s |
| air-cart-fp16384 | 2 | 7.71 s (7.65 s–7.76 s) | 711 ms | 7.00 s | 0 ms | 7.6 s |
| air-cart-fp65536 | 2 | 7.35 s (7.33 s–7.38 s) | 706 ms | 6.65 s | 0 ms | 2.6 s |
| air-cart-fp16384-tr16384 | 2 | 7.57 s (7.54 s–7.59 s) | 669 ms | 6.90 s | 0 ms | 8.4 s |
| air-c45-fp8192 | 2 | 16.66 s (16.56 s–16.76 s) | 730 ms | 12.25 s | 3.68 s | 32.3 s |
| air-c45-fp16384 | 2 | 15.79 s (15.77 s–15.82 s) | 743 ms | 11.29 s | 3.77 s | 11.2 s |
| air-c45-fp65536 | 2 | 15.14 s (15.07 s–15.20 s) | 709 ms | 10.68 s | 3.75 s | 7.9 s |
| air-c45-fp16384-tr16384 | 2 | 15.55 s (15.54 s–15.56 s) | 704 ms | 11.13 s | 3.72 s | 11.1 s |

#### 12-epyc-both-feature-parallel.csv

`epyc1, 48 CPUs, job 192, 2026-10-14 15:29:30, load 3.79 13.03 9.81`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| higgs-cart-fp16384 | 3 | 1.07 s (1.05 s–1.07 s) | 108 ms | 947 ms | 1 ms | 2.0 s |
| higgs-cart-fp32768 | 3 | 1.04 s (1.03 s–1.13 s) | 115 ms | 929 ms | 1 ms | 1.0 s |
| higgs-cart-fp65536 | 3 | 1.08 s (1.07 s–1.12 s) | 117 ms | 958 ms | 1 ms | 0.8 s |
| higgs-c45-fp16384 | 3 | 1.93 s (1.89 s–1.93 s) | 132 ms | 1.47 s | 327 ms | 1.5 s |
| higgs-c45-fp32768 | 3 | 1.93 s (1.91 s–1.99 s) | 130 ms | 1.48 s | 329 ms | 1.2 s |
| higgs-c45-fp65536 | 3 | 2.01 s (1.98 s–2.04 s) | 144 ms | 1.53 s | 326 ms | 1.2 s |
| airline-cart-fp16384 | 3 | 7.64 s (7.63 s–7.67 s) | 698 ms | 6.96 s | 0 ms | 6.8 s |
| airline-cart-fp32768 | 3 | 7.33 s (7.27 s–7.34 s) | 673 ms | 6.66 s | 0 ms | 3.3 s |
| airline-cart-fp65536 | 3 | 7.28 s (7.27 s–7.38 s) | 661 ms | 6.63 s | 0 ms | 2.6 s |
| airline-c45-fp16384 | 3 | 15.69 s (15.67 s–15.72 s) | 698 ms | 11.23 s | 3.74 s | 10.7 s |
| airline-c45-fp32768 | 3 | 15.18 s (15.09 s–15.20 s) | 710 ms | 10.73 s | 3.69 s | 7.6 s |
| airline-c45-fp65536 | 3 | 15.05 s (14.88 s–15.14 s) | 719 ms | 10.64 s | 3.72 s | 7.9 s |

#### 12-hopper-both-min-rows.csv

`hopper, 32 CPUs, job 191, 2026-10-14 15:30:10, load 1.57 4.98 5.68`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| higgs-cart-min1024 | 3 | 436 ms (422 ms–486 ms) | 20 ms | 320 ms | 1 ms | 2.9 s |
| higgs-cart-min4096 | 3 | 468 ms (437 ms–468 ms) | 20 ms | 348 ms | 1 ms | 2.8 s |
| higgs-cart-min16384 | 3 | 658 ms (579 ms–673 ms) | 20 ms | 538 ms | 1 ms | 2.7 s |
| higgs-c45-min1024 | 3 | 1.90 s (1.86 s–1.93 s) | 112 ms | 735 ms | 955 ms | 3.8 s |
| higgs-c45-min4096 | 3 | 1.88 s (1.88 s–1.88 s) | 113 ms | 734 ms | 918 ms | 4.1 s |
| higgs-c45-min16384 | 3 | 2.11 s (2.11 s–2.15 s) | 113 ms | 942 ms | 968 ms | 3.9 s |
| airline-cart-min1024 | 3 | 2.08 s (2.07 s–2.09 s) | 87 ms | 1.64 s | 1 ms | 8.4 s |
| airline-cart-min4096 | 3 | 2.18 s (2.17 s–2.19 s) | 87 ms | 1.73 s | 1 ms | 8.1 s |
| airline-cart-min16384 | 3 | 2.21 s (2.20 s–2.23 s) | 87 ms | 1.76 s | 1 ms | 7.8 s |
| airline-c45-min1024 | 3 | 21.80 s (21.35 s–22.91 s) | 518 ms | 12.26 s | 8.70 s | 18.3 s |
| airline-c45-min4096 | 3 | 21.49 s (20.99 s–21.91 s) | 515 ms | 10.84 s | 9.78 s | 16.4 s |
| airline-c45-min16384 | 3 | 18.36 s (18.06 s–19.93 s) | 517 ms | 8.40 s | 9.77 s | 15.7 s |

#### 13-epyc-airline-prune-breakdown.csv

`epyc1, 48 CPUs, job 194, 2026-10-14 15:33:55, load 37.17 26.06 16.04`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| air-c45-par | 1 | 15.82 s (15.82 s–15.82 s) | 765 ms | 11.32 s | 3.73 s | 11.8 s |

#### 13-hopper-airline-prune-breakdown.csv

`hopper, 32 CPUs, job 195, 2026-10-14 15:35:22, load 14.01 11.71 8.47`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| air-c45-cuda | 1 | 21.92 s (21.92 s–21.92 s) | 524 ms | 11.89 s | 9.14 s | 18.4 s |
| air-c45-par | 1 | 24.84 s (24.84 s–24.84 s) | 2.12 s | 14.01 s | 8.71 s | 15.2 s |

#### 13-hopper-min-rows-fine.csv

`hopper, 32 CPUs, job 196, 2026-10-14 15:36:53, load 10.23 11.56 8.74`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| airline-c45-min8192 | 3 | 19.88 s (19.45 s–20.33 s) | 513 ms | 9.99 s | 9.18 s | 16.7 s |
| airline-c45-min16384 | 3 | 17.53 s (17.23 s–17.80 s) | 515 ms | 7.83 s | 8.90 s | 15.2 s |
| airline-c45-min32768 | 3 | 17.59 s (16.60 s–18.34 s) | 513 ms | 7.42 s | 9.28 s | 17.3 s |
| airline-c45-min65536 | 3 | 17.22 s (17.17 s–18.46 s) | 520 ms | 7.33 s | 9.01 s | 17.8 s |
| airline-cart-min8192 | 3 | 2.23 s (2.18 s–2.25 s) | 87 ms | 1.76 s | 1 ms | 7.7 s |
| airline-cart-min16384 | 3 | 2.18 s (2.15 s–2.23 s) | 87 ms | 1.76 s | 1 ms | 7.0 s |
| higgs-cart-min1024 | 3 | 436 ms (412 ms–436 ms) | 20 ms | 316 ms | 1 ms | 3.0 s |
| higgs-c45-min1024 | 3 | 1.88 s (1.83 s–1.92 s) | 115 ms | 715 ms | 958 ms | 3.8 s |
| higgs-cart-min1536 | 3 | 434 ms (431 ms–510 ms) | 20 ms | 315 ms | 1 ms | 3.0 s |
| higgs-c45-min1536 | 3 | 1.87 s (1.85 s–2.05 s) | 134 ms | 722 ms | 916 ms | 4.2 s |
| higgs-cart-min2048 | 3 | 444 ms (440 ms–446 ms) | 20 ms | 326 ms | 1 ms | 3.0 s |
| higgs-c45-min2048 | 3 | 1.87 s (1.82 s–1.92 s) | 112 ms | 726 ms | 935 ms | 3.9 s |

#### 14-epyc-final-old-vs-new.csv

`epyc1, 48 CPUs, job 200, 2026-10-14 15:45:03, load 0.00 3.03 8.18`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| higgs-cart-old | 3 | 1.95 s (1.95 s–2.09 s) | 212 ms | 1.72 s | 12 ms | 25.8 s |
| higgs-cart-new | 3 | 1.07 s (1.05 s–1.08 s) | 123 ms | 937 ms | 1 ms | 1.1 s |
| higgs-c45-old | 3 | 2.55 s (2.51 s–2.60 s) | 247 ms | 1.84 s | 455 ms | 21.3 s |
| higgs-c45-new | 3 | 1.86 s (1.82 s–1.89 s) | 133 ms | 1.47 s | 278 ms | 1.3 s |
| airline-cart-old | 3 | 15.98 s (15.08 s–16.13 s) | 1.86 s | 14.00 s | 123 ms | 201.2 s |
| airline-cart-new | 3 | 7.34 s (7.28 s–7.36 s) | 683 ms | 6.64 s | 1 ms | 3.5 s |
| airline-c45-old | 3 | 24.67 s (23.94 s–24.79 s) | 2.06 s | 17.16 s | 5.49 s | 221.9 s |
| airline-c45-new | 3 | 13.93 s (13.90 s–13.96 s) | 675 ms | 10.33 s | 2.93 s | 8.2 s |

#### 14-hopper-final-old-vs-new.csv

`hopper, 32 CPUs, job 199, 2026-10-14 15:52:07, load 13.01 9.88 9.44`

| case | runs | train (median, min–max) | presort | build | prune | sys CPU |
|---|--:|--:|--:|--:|--:|--:|
| higgs-cart-cuda-old | 3 | 1.48 s (1.45 s–1.50 s) | 20 ms | 1.27 s | 19 ms | 7.0 s |
| higgs-cart-cuda-new | 3 | 536 ms (445 ms–537 ms) | 20 ms | 397 ms | 1 ms | 3.2 s |
| higgs-c45-cuda-old | 3 | 2.61 s (2.54 s–2.61 s) | 158 ms | 974 ms | 1.30 s | 11.4 s |
| higgs-c45-cuda-new | 3 | 1.65 s (1.63 s–1.69 s) | 113 ms | 668 ms | 771 ms | 3.8 s |
| airline-cart-cuda-old | 3 | 11.64 s (11.37 s–11.76 s) | 87 ms | 10.64 s | 235 ms | 55.0 s |
| airline-cart-cuda-new | 3 | 2.32 s (2.31 s–2.34 s) | 87 ms | 1.86 s | 1 ms | 7.4 s |
| airline-c45-cuda-old | 3 | 26.03 s (25.97 s–27.83 s) | 628 ms | 10.55 s | 14.15 s | 39.1 s |
| airline-c45-cuda-new | 3 | 14.53 s (13.49 s–14.67 s) | 523 ms | 6.09 s | 7.43 s | 14.0 s |
