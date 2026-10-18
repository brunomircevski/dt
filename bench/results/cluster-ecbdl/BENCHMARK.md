# ECBDL'14 on the mordor3 cluster: a 15M-row sample on two GPU and three CPU servers, the full 32M rows on the CPU servers

**Status:** one run per case, all on 2026-10-17, commit cd18717 (one copy of the presorted columns on every backend, [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)). `machine.json` says `-dirty` only because the previous result files had been removed from the cluster's checkout for the rerun; the code was committed.
- **15M sample** (`*-ecbdl_15m`, pruned and `unpruned/`): hopper and ampere 18:06–18:21, dell2 18:16–18:30, epyc1 18:49–19:01, dell1 19:24–19:36.
- **Full set** (`*-ecbdl`, pruned and `unpruned/`): dell2 17:48–18:16, epyc1 19:01–19:27, dell1 19:36–19:59.
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

**dell2 is not dell1 with less RAM:** its 8 modules fill half of the CPUs' 16 memory channels (8 per socket), so it has about half of dell1's memory bandwidth, and ./tree is bound by memory bandwidth. It is 1.4–1.55× slower than dell1 on the sample and 1.35–1.45× on the full set.

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the CSV). Three protocols, one section each:
- `cart_alpha_nodepth`: CART pruned at the fixed α = 1e-5, no depth limit.
- `cart_full_nodepth`: CART grown to pure leaves, no pruning, no depth limit.
- `c45`: C4.5, CF 0.25. A bar's lighter part is the pruning.

## 15M-row sample

![ECBDL'14 15M sample on five mordor3 nodes](chart.png)

| | CART | CART unpruned | C4.5 (of which pruning) | RAM CART / unpruned / C4.5 | VRAM CART / unpruned / C4.5 |
|---|--:|--:|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **7.97 s** | **18.39 s** | **45.58 s** (17.65 s) | 39.8 / 106.4 / 107.3 GiB | 73.4 / 73.6 / 73.5 GiB |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 11.61 s | 25.38 s | 66.93 s (28.23 s) | 39.8 / 106.4 / 107.6 GiB | 73.3 / 73.6 / 73.5 GiB |
| Parallel · epyc (EPYC 9255, 48 threads) | 15.96 s | 34.31 s | 76.11 s (10.12 s) | 111.4 / 111.5 / 112.8 GiB | – |
| Parallel · dell (Xeon 6542Y, 48 threads) | 24.82 s | 51.91 s | 115.64 s (16.93 s) | 111.5 / 111.5 / 112.8 GiB | – |
| Parallel · dell512 (the same, half the memory channels) | 38.05 s | 80.43 s | 161.11 s (30.71 s) | 111.4 / 111.5 / 112.8 GiB | – |

- **The H100 is 2.0× (CART), 1.9× (unpruned CART) and 1.7× (C4.5) faster than epyc**, a little more than on the 6M sample (1.8×, 1.7× and 1.6×), still less than on airline (6.4× for pruned CART). Profiled in [cluster-ecbdl-profile](../cluster-ecbdl-profile/BENCHMARK.md) (2026-10-18): the H100 grows 2.0–2.5× faster than epyc, but C4.5's pruning runs on the slower host CPUs, and the data's upload and the page-locking of the hand-off arena cost 0.7–1.1 s on hopper.
- **C4.5's pruning is 39–42% of a CUDA run** (27–31% on the 6M sample) and slower than on the CPU servers (17.7 and 28.2 s against epyc's 10.1 s). It runs on the GPU nodes' slower host CPUs, and its first step (the rows sorted by leaf) runs on the host too: a copy of the raw values for doing it on the GPU (4 bytes per value, 35 GiB) no longer fits beside the columns, and ./tree keeps it only when it fits.
- **VRAM: 73.3–73.6 GiB of the 80 GB** (79.6 GiB usable on the H100). The columns take 70.6 GiB: 632 slots (631 features and one spare) of 15M rows at 8 bytes; the rest is the 512 MB hand-off ring, the per-level arrays and the CUDA context.
- **RAM:** the CPU servers hold the loaded float32 data (35 GiB) and the columns in 631 + 47 slots (76 GiB): 111–113 GiB. A CUDA run keeps 39.8 GiB on the host for pruned CART (the loaded data and the CPU's share of small subtrees), about 107 GiB for C4.5 and unpruned CART, whose many small subtrees are grown on the host from a host copy of the columns.

## Full set (31.9M rows), CPU servers

![full ECBDL'14 on epyc, dell and dell512](chart-full.png)

| | CART | CART unpruned | C4.5 (of which pruning) | RAM CART / unpruned / C4.5 |
|---|--:|--:|--:|--:|
| Parallel · epyc (EPYC 9255, 48 threads) | **35.44 s** | **84.30 s** | **220.75 s** (35.94 s) | 236.9 / 237.0 / 238.9 GiB |
| Parallel · dell (Xeon 6542Y, 48 threads) | 52.10 s | 121.92 s | 336.65 s (40.00 s) | 236.9 / 237.0 / 238.9 GiB |
| Parallel · dell512 (the same, half the memory channels) | 72.81 s | 176.32 s | 458.63 s (97.76 s) | 236.9 / 237.0 / 238.8 GiB |

- **About 12.6 bytes per value of RAM** (254 GB): 4 for the loaded data and 8 × 678/631 for the columns in 631 + 47 slots (min(features − 1, threads) spare ones). It was 20 bytes per value (375.5–377.4 GiB) with the second copy of every column; 37% less.
- **epyc is 1.5× faster than dell** on every case (CART 35.4 against 52.1 s, unpruned CART 84.3 against 121.9 s, C4.5 220.8 against 336.7 s).
- **2.1× the sample's rows:** unpruned CART takes 2.5× (epyc) and 2.3× (dell) the sample's time, C4.5 2.9× on both; their trees have 2.0× and 2.2× the nodes.
- **Against the previous run** (8da0c12, 2026-10-17 00:29–10:35, the same trees): dell within 1–2% (C4.5 338.5 → 336.7 s, unpruned CART 123.0 → 121.9 s); dell512 −7% to +2% (CART 78.3 → 72.8 s, C4.5 452.4 → 458.6 s); epyc's pruned CART 51.6 → 35.4 s, its presort no longer 21 s (below), unpruned CART 83.6 → 84.3 s and C4.5 209.8 → 220.8 s (+5%, mostly the pruning, 27.4 → 35.9 s, in one run each; not looked into).

### Filling the memory of a 512 GB node

The presort is where ./tree first touches its memory. Presort time (and kernel time, `sys`, all threads together):

| Presort (sys) | pruned CART | C4.5 | unpruned CART |
|---|--:|--:|--:|
| epyc (512 GB) | 6.34 s (203 s) | 4.79 s (121 s) | 5.51 s (168 s) |
| dell512 (512 GB, half the channels) | 9.45 s (200 s) | 12.05 s (206 s) | 9.62 s (229 s) |
| dell (1 TB) | 5.05 s (115 s) | 6.38 s (161 s) | 5.06 s (133 s) |

- **epyc's slow presorts are gone.** With 375 GiB to fill, epyc's presort took 5.4–21.2 s (191–972 s of `sys`), most likely because the page cache of the data files (81 GB `train.f32`, 59 GB `train.csv`, read before each process) and ./tree's 375 GiB did not fit side by side, so the kernel dropped cache pages while ./tree's threads faulted in their memory. With 237 GiB they fit, and epyc presorts as fast as dell.
- **dell512's presort is still about 1.8–1.9× dell's,** on the full set and on the 15M sample alike (4.4–5.4 s against 2.4–2.9 s), where nothing comes near 512 GB. So its half memory bandwidth, not the reclaim, explains most of it. On the 6M sample the gap was 1.4–1.5×.

## The trees

- **Pruned CART is a single leaf:** at α = 1e-5 every subtree costs more than it saves (with 2.15% contacts a split must fix more than α · n rows per extra leaf), so the pruned tree predicts "no contact" everywhere (training accuracy 97.85%, the majority share). With a fixed α ./tree does not split nodes that misclassify fewer than α · n rows ([docs/ALGORITHMS.md](../../../docs/ALGORITHMS.md)), so the tree it grows before pruning is much smaller than the unpruned one.
- **Unpruned CART:** 604,225 nodes, depth 94 (sample); 1,205,683 nodes, depth 111 (full); training accuracy 100%.
- **C4.5:** 209,799 nodes, depth 266, training accuracy 99.09% (sample); 470,673 nodes, depth 327, 99.19% (full).
