# ECBDL'14 on the mordor3 cluster: a 15M-row sample on two GPU and three CPU servers, the full 32M rows on the CPU servers

**Status:** one run per case, all on 2026-10-19, commit 051949b: cd18717 (one copy of the presorted columns on every backend, [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)) plus `./tree --profile` and the count tables filled without a zero-fill on one thread. The same trees as on cd18717, the times within −2% to +5%. `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **15M sample** (`*-ecbdl_15m`, pruned and `unpruned/`): dell2 15:10–15:22, epyc1 15:14–15:25, dell1 15:15–15:25.
- **CUDA rows rerun at 6f6cecb** (hopper and ampere, 2026-10-19 20:25–20:40): the GPU's hand-offs go through page-locked buffers instead of a page-locked arena ([cluster-handoff-staging](../cluster-handoff-staging/BENCHMARK.md)), and the host threads partition big nodes in blocks (d9da007). The same trees. 051949b's CUDA times are given as "was" below.
- **Full set** (`*-ecbdl`, pruned and `unpruned/`): dell2 15:23–15:51, epyc1 15:25–15:51, dell1 15:25–15:50.
- **Each node ran alone** and was used whole. The jobs ran one after another on each node, after the other cluster benchmarks of the same commit.
- **Same trees:** every case built the same tree on every node and backend, and the full set's trees are the same as before the rerun.
- **Settings:** ./tree's defaults, as in [cluster-airlines](../cluster-airlines/BENCHMARK.md). Only training time and memory are compared; accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).
- **The sample grew from 6M to 15M rows:** the GPU now needs 8 bytes per value instead of 16, so 15M rows fill the 80 GB GPUs about as much as 6M did before. The 6M results (commit 8da0c12 and 6c7a43b, the night before) are in the git history up to commit cd18717.

How the runs are made, and the dataset: [bench/cluster/README.md](../../cluster/README.md).

```bash
P="-p cart_full_nodepth -o cluster-ecbdl/unpruned"
bench/cluster/submit.sh -d ecbdl_15m -a <prepare job>    # pruned, four nodes
bench/cluster/submit.sh -d ecbdl                         # pruned, full set: epyc1 and dell1
bench/cluster/submit.sh -d ecbdl_15m -a <prepare job> $P # unpruned CART
bench/cluster/submit.sh -d ecbdl $P
bench/cluster/submit.sh -d ecbdl_15m cpu-dell512-ecbdl_15m     # dell2, by name
bench/cluster/submit.sh -d ecbdl cpu-dell512-ecbdl
bench/cluster/submit.sh -d ecbdl_15m $P cpu-dell512-ecbdl_15m
bench/cluster/submit.sh -d ecbdl $P cpu-dell512-ecbdl
d=bench/results/cluster-ecbdl                            # the charts
bench/.venv/bin/python bench/machines_chart.py --protocols cart_alpha_nodepth,cart_full_nodepth,c45 \
    --more $d/unpruned $d/cuda-hopper-ecbdl_15m $d/cuda-ampere-ecbdl_15m $d/cpu-epyc-ecbdl_15m \
    $d/cpu-dell-ecbdl_15m $d/cpu-dell512-ecbdl_15m --out $d/chart.png
bench/.venv/bin/python bench/machines_chart.py --protocols cart_alpha_nodepth,cart_full_nodepth,c45 \
    --more $d/unpruned $d/cpu-epyc-ecbdl $d/cpu-dell-ecbdl $d/cpu-dell512-ecbdl --out $d/chart-full.png
```

The dataset is ECBDL'14 (protein contact-map prediction, the GECCO 2014 big data competition): 631 features (539 real, 92 nominal as integer codes), label 0/1 with 2.15% contacts. The full set has 31,892,921 training rows, 20.1 G values (65× HIGGS's, 81 GB as float32). The CUDA backend needs 8 bytes per value on the GPU (one copy of the presorted columns, plus one spare column), about 160 GB for the full set, so the GPUs get `ecbdl_15m`, a random 15M of those rows (9.5 G values; the first 15M of the shuffled training rows, so the earlier 6M sample is part of it). The CPU servers run both.

| Run | Node | CPU | RAM | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|---|
| `cuda-hopper-ecbdl_15m` | hopper | 2× Xeon Gold 6444Y, 32 cores | 1 TB DDR5-4800 | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-ecbdl_15m` | ampere | 2× EPYC 7402, 48 cores | 1 TB DDR4-3200 | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-*` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – | parallel (48 threads) |
| `cpu-dell-*` | dell1 | 2× Xeon Gold 6542Y, 48 cores | 1 TB DDR5-5200 (16 × 64 GB) | – | parallel (48 threads) |
| `cpu-dell512-*` | dell2 | the same as dell1 | 512 GB DDR5-5200 (8 × 64 GB) | – | parallel (48 threads) |

**dell2 is not dell1 with less RAM:** its 8 modules fill half of the CPUs' 16 memory channels (8 per socket), so it has about half of dell1's memory bandwidth, and ./tree is bound by memory bandwidth. It is 1.45–1.55× slower than dell1 on the sample and 1.35–1.45× on the full set.

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the CSV). Three protocols, one section each:
- `cart_alpha_nodepth`: CART pruned at the fixed α = 1e-5, no depth limit.
- `cart_full_nodepth`: CART grown to pure leaves, no pruning, no depth limit.
- `c45`: C4.5, CF 0.25. A bar's lighter part is the pruning.

## 15M-row sample

![ECBDL'14 15M sample on five mordor3 nodes](chart.png)

| | CART | CART unpruned | C4.5 (of which pruning) | RAM CART / unpruned / C4.5 | VRAM CART / unpruned / C4.5 |
|---|--:|--:|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **7.96 s** (was 8.00) | **17.35 s** (was 18.28) | **50.70 s** (24.00 s; was 46.56) | 37.4 / 106.6 / 107.6 GiB | 73.4 / 73.5 / 73.4 GiB |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 11.67 s (was 11.68) | 23.69 s (was 25.93) | 65.53 s (28.67 s; was 66.95) | 37.4 / 106.6 / 107.8 GiB | 73.3 / 73.4 / 73.4 GiB |
| Parallel · epyc (EPYC 9255, 48 threads) | 16.69 s | 35.78 s | 77.95 s (10.86 s) | 111.4 / 111.5 / 112.8 GiB | – |
| Parallel · dell (Xeon 6542Y, 48 threads) | 24.85 s | 51.74 s | 113.52 s (14.53 s) | 111.4 / 111.5 / 112.8 GiB | – |
| Parallel · dell512 (the same, half the memory channels) | 38.50 s | 80.08 s | 163.59 s (31.88 s) | 111.4 / 111.5 / 112.8 GiB | – |

- **The H100 is 2.1× (CART), 2.1× (unpruned CART) and 1.5× (C4.5) faster than epyc**, a little more than on the 6M sample (1.8×, 1.7× and 1.6×) except C4.5, still less than on airline (6.2× for pruned CART). Profiled in [cluster-ecbdl-profile](../cluster-ecbdl-profile/BENCHMARK.md): the H100 grows 2.0–2.5× faster than epyc, but C4.5's pruning runs on the slower host CPUs.
- **6f6cecb against 051949b on the GPUs:** the page-locking of the 75 GB hand-off arena is gone (it held up the GPU loop 1.2–1.9 s): unpruned CART 18.28 → 17.35 s (H100) and 25.93 → 23.69 s (A100), A100 C4.5 66.95 → 65.53 s; pruned CART, which hands over little, is unchanged. The H100's C4.5 went 46.56 → 50.70 s, all in the pruning on the host CPUs (18.7 → 24.0 s; growing 27.9 → 26.7 s). Pruning does not touch the changed code: in a same-job comparison that evening both builds pruned equally slowly (52.7 s against 51.3 s in total), with hopper's NUMA node 0 nearly full (11–28 GB free of 516).
- **C4.5's pruning is 44–47% of a CUDA run** (27–31% on the 6M sample) and slower than on the CPU servers (24.0 and 28.7 s against epyc's 10.9 s). It runs on the GPU nodes' slower host CPUs, and its first step (the rows sorted by leaf) runs on the host too: a copy of the raw values for doing it on the GPU (4 bytes per value, 35 GiB) no longer fits beside the columns, and ./tree keeps it only when it fits.
- **VRAM: 73.3–73.5 GiB of the 80 GB** (79.6 GiB usable on the H100). The columns take 70.6 GiB: 632 slots (631 features and one spare) of 15M rows at 8 bytes; the rest is the 512 MB hand-off ring, the per-level arrays and the CUDA context.
- **RAM:** the CPU servers hold the loaded float32 data (35 GiB) and the columns in 631 + 47 slots (76 GiB): 111–113 GiB. A CUDA run keeps 37.4 GiB on the host for pruned CART (the loaded data and the CPU's share of small subtrees; 39.8 GiB at 051949b, which touched the whole hand-off arena), about 107 GiB for C4.5 and unpruned CART, whose many small subtrees are grown on the host from a host copy of the columns.

## Full set (31.9M rows), CPU servers

![full ECBDL'14 on epyc, dell and dell512](chart-full.png)

| | CART | CART unpruned | C4.5 (of which pruning) | RAM CART / unpruned / C4.5 |
|---|--:|--:|--:|--:|
| Parallel · epyc (EPYC 9255, 48 threads) | **35.46 s** | **83.40 s** | **220.88 s** (35.91 s) | 236.9 / 237.0 / 238.9 GiB |
| Parallel · dell (Xeon 6542Y, 48 threads) | 51.89 s | 121.93 s | 337.94 s (40.11 s) | 236.9 / 237.0 / 238.9 GiB |
| Parallel · dell512 (the same, half the memory channels) | 73.36 s | 176.70 s | 458.24 s (86.26 s) | 236.9 / 237.0 / 238.8 GiB |

- **About 12.6 bytes per value of RAM** (254 GB): 4 for the loaded data and 8 × 678/631 for the columns in 631 + 47 slots (min(features − 1, threads) spare ones). It was 20 bytes per value (375.5–377.4 GiB) with the second copy of every column; 37% less.
- **epyc is 1.5× faster than dell** on every case (CART 35.5 against 51.9 s, unpruned CART 83.4 against 121.9 s, C4.5 220.9 against 337.9 s).
- **2.1× the sample's rows:** unpruned CART takes 2.3× (epyc) and 2.4× (dell) the sample's time, C4.5 2.8× and 3.0×; their trees have 2.0× and 2.2× the nodes.
- **051949b against cd18717** (2026-10-17, the same trees): within −2% to +2.2% on the full set, dell's and dell2's sample and the GPUs (largest: H100 C4.5 45.58 → 46.56 s, A100 unpruned CART 25.38 → 25.93 s); epyc's sample +2% to +5% (CART 15.96 → 16.69 s, unpruned 34.31 → 35.78 s, C4.5 76.11 → 77.95 s), in one run each. Only `--profile` (off here) and the count tables' fill changed.
- **cd18717 against 8da0c12** (2026-10-17 00:29–10:35, the same trees): dell within 1–2% (C4.5 338.5 → 336.7 s, unpruned CART 123.0 → 121.9 s); dell512 −7% to +2% (CART 78.3 → 72.8 s, C4.5 452.4 → 458.6 s); epyc's pruned CART 51.6 → 35.4 s, its presort no longer 21 s (below), unpruned CART 83.6 → 84.3 s and C4.5 209.8 → 220.8 s (+5%, mostly the pruning, 27.4 → 35.9 s, in one run each; not looked into).

### Filling the memory of a 512 GB node

The presort is where ./tree first touches its memory. Presort time (and kernel time, `sys`, all threads together):

| Presort (sys) | pruned CART | C4.5 | unpruned CART |
|---|--:|--:|--:|
| epyc (512 GB) | 5.70 s (181 s) | 5.00 s (139 s) | 5.26 s (142 s) |
| dell512 (512 GB, half the channels) | 10.18 s (229 s) | 12.79 s (258 s) | 9.93 s (254 s) |
| dell (1 TB) | 4.99 s (83 s) | 6.35 s (160 s) | 5.06 s (124 s) |

- **epyc's slow presorts are gone.** With 375 GiB to fill, epyc's presort took 5.4–21.2 s (191–972 s of `sys`), most likely because the page cache of the data files (81 GB `train.f32`, 59 GB `train.csv`, read before each process) and ./tree's 375 GiB did not fit side by side, so the kernel dropped cache pages while ./tree's threads faulted in their memory. With 237 GiB they fit, and epyc presorts as fast as dell.
- **dell512's presort is still about 1.8–2.0× dell's,** on the full set and on the 15M sample alike (4.2–5.6 s against 2.4–2.9 s), where nothing comes near 512 GB. So its half memory bandwidth, not the reclaim, explains most of it. On the 6M sample the gap was 1.4–1.5×.

## The trees

- **Pruned CART is a single leaf:** at α = 1e-5 every subtree costs more than it saves (with 2.15% contacts a split must fix more than α · n rows per extra leaf), so the pruned tree predicts "no contact" everywhere (training accuracy 97.85%, the majority share). With a fixed α ./tree does not split nodes that misclassify fewer than α · n rows ([docs/ALGORITHMS.md](../../../docs/ALGORITHMS.md)), so the tree it grows before pruning is much smaller than the unpruned one.
- **Unpruned CART:** 604,225 nodes, depth 94 (sample); 1,205,683 nodes, depth 111 (full); training accuracy 100%.
- **C4.5:** 209,799 nodes, depth 266, training accuracy 99.09% (sample); 470,673 nodes, depth 327, 99.19% (full).
