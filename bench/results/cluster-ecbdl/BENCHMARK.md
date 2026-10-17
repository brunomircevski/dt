# ECBDL'14 on the mordor3 cluster: a 6M-row sample on two GPU and three CPU servers, the full 32M rows on the CPU servers

**Status:** one run per case, all on 2026-10-17, with the ./tree code and binaries of 8da0c12:
- **Pruned CART and C4.5** (the run directories here) on hopper, ampere, epyc1 and dell1: 00:06–01:07, commit 49ebb6d.
- **Unpruned CART** (`unpruned/`) on the same nodes: 09:49–10:03, commit 49ebb6d.
- **dell2** ("dell512", both sets): 09:59–10:35, commit 6c7a43b. These jobs went in through `job.sh` with the options of `submit.sh`, which got the `cpu-dell512-<data>` entry afterwards.
- **Each node ran alone** and was used whole. The first jobs started when the data was prepared (SLURM dependency), the CPU jobs right after the SUSY-vs-others jobs on the same nodes.
- **Same trees:** every case built the same tree on every node and backend.
- **Settings:** ./tree's defaults, as in [cluster-airlines](../cluster-airlines/BENCHMARK.md). Only training time and memory are compared; accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).

How the runs are made, and the dataset: [bench/cluster/README.md](../../cluster/README.md).

```bash
P="-p cart_full_nodepth -o cluster-ecbdl/unpruned"
bench/cluster/submit.sh -d ecbdl_6m -a <prepare job>     # pruned, four nodes
bench/cluster/submit.sh -d ecbdl -a <prepare job>        # pruned, full set: epyc1 and dell1
bench/cluster/submit.sh -d ecbdl_6m $P                   # unpruned CART
bench/cluster/submit.sh -d ecbdl $P
bench/cluster/submit.sh -d ecbdl_6m cpu-dell512-ecbdl_6m       # dell2, by name
bench/cluster/submit.sh -d ecbdl cpu-dell512-ecbdl
bench/cluster/submit.sh -d ecbdl_6m $P cpu-dell512-ecbdl_6m
bench/cluster/submit.sh -d ecbdl $P cpu-dell512-ecbdl
d=bench/results/cluster-ecbdl                            # the charts
bench/.venv/bin/python bench/machines_chart.py --protocols cart_alpha_nodepth,cart_full_nodepth,c45 \
    --more $d/unpruned $d/cuda-hopper-ecbdl_6m $d/cuda-ampere-ecbdl_6m $d/cpu-epyc-ecbdl_6m \
    $d/cpu-dell-ecbdl_6m $d/cpu-dell512-ecbdl_6m --out $d/chart.png
bench/.venv/bin/python bench/machines_chart.py --protocols cart_alpha_nodepth,cart_full_nodepth,c45 \
    --more $d/unpruned $d/cpu-epyc-ecbdl $d/cpu-dell-ecbdl $d/cpu-dell512-ecbdl --out $d/chart-full.png
```

The dataset is ECBDL'14 (protein contact-map prediction, the GECCO 2014 big data competition): 631 features (539 real, 92 nominal as integer codes), label 0/1 with 2.15% contacts. The full set has 31,892,921 training rows, 20.1 G values (65× HIGGS's, 81 GB as float32). The CUDA backend needs 16 bytes per value on the GPU, about 320 GB for the full set, so the GPUs get `ecbdl_6m`, a random 6M of those rows (3.8 G values). The CPU servers run both.

| Run | Node | CPU | RAM | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|---|
| `cuda-hopper-ecbdl_6m` | hopper | 2× Xeon Gold 6444Y, 32 cores | 1 TB DDR5-4800 | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-ecbdl_6m` | ampere | 2× EPYC 7402, 48 cores | 1 TB DDR4-3200 | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-*` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – | parallel (48 threads) |
| `cpu-dell-*` | dell1 | 2× Xeon Gold 6542Y, 48 cores | 1 TB DDR5-5200 (16 × 64 GB) | – | parallel (48 threads) |
| `cpu-dell512-*` | dell2 | the same as dell1 | 512 GB DDR5-5200 (8 × 64 GB) | – | parallel (48 threads) |

**dell2 is not dell1 with less RAM:** its 8 modules fill half of the CPUs' 16 memory channels (8 per socket), so it has about half of dell1's memory bandwidth, and ./tree is bound by memory bandwidth. It is 1.25–1.5× slower than dell1 in every case, the 6M sample included.

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the CSV). Three protocols, one section each:
- `cart_alpha_nodepth`: CART pruned at the fixed α = 1e-5, no depth limit.
- `cart_full_nodepth`: CART grown to pure leaves, no pruning, no depth limit.
- `c45`: C4.5, CF 0.25. A bar's lighter part is the pruning.

## 6M-row sample

![ECBDL'14 6M sample on five mordor3 nodes](chart.png)

| | CART | CART unpruned | C4.5 (of which pruning) | RAM CART / unpruned / C4.5 | VRAM CART / unpruned / C4.5 |
|---|--:|--:|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **4.08 s** | **7.62 s** | **13.92 s** (3.78 s) | 17.6 / 42.8 / 43.0 GiB | 57.3 / 57.4 / 71.5 GiB |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 5.77 s | 11.00 s | 20.46 s (6.43 s) | 17.6 / 42.8 / 43.1 GiB | 57.2 / 57.4 / 71.4 GiB |
| Parallel · epyc (EPYC 9255, 48 threads) | 7.35 s | 12.94 s | 22.70 s (2.00 s) | 70.7 / 70.7 / 71.3 GiB | – |
| Parallel · dell (Xeon 6542Y, 48 threads) | 11.97 s | 19.44 s | 34.46 s (3.33 s) | 70.7 / 70.7 / 71.3 GiB | – |
| Parallel · dell512 (the same, half the memory channels) | 15.79 s | 25.68 s | 42.97 s (5.95 s) | 70.7 / 70.7 / 71.2 GiB | – |

- **The H100 is 1.8× (CART), 1.6× (C4.5) and 1.7× (unpruned CART) faster than epyc**, less than on airline (6.6× for pruned CART): the GPU has no room for the mirror of the CPU hand-off (another 8 bytes per value), so its copies to the host no longer overlap the next levels, and C4.5's pruning on the host CPU is 27–31% of a CUDA run.
- **VRAM is as computed from the allocations:** two copies of the presorted columns (16 bytes per value, 56.4 GiB) plus the CUDA context and per-level arrays; C4.5 also keeps the raw values for its pruning (4 bytes per value, 14.1 GiB). C4.5 uses 71.5 of the 80 GB.
- **The CPU servers hold the same columns in RAM** (8 bytes per value, twice for the out-of-place partition, plus the loaded float32 data): 70.7 GiB. A CUDA run keeps 17.6 GiB on the host for pruned CART (the loaded data and the CPU's share of small subtrees), about 43 GiB for C4.5 and unpruned CART, whose many small subtrees are grown on the host.

## Full set (31.9M rows), CPU servers

![full ECBDL'14 on epyc, dell and dell512](chart-full.png)

| | CART | CART unpruned | C4.5 (of which pruning) | RAM CART / unpruned / C4.5 |
|---|--:|--:|--:|--:|
| Parallel · epyc (EPYC 9255, 48 threads) | **51.57 s** | **83.58 s** | **209.79 s** (27.36 s) | 375.5 / 375.5 / 377.4 GiB |
| Parallel · dell (Xeon 6542Y, 48 threads) | 52.85 s | 123.00 s | 338.54 s (40.12 s) | 375.5 / 375.5 / 377.4 GiB |
| Parallel · dell512 (the same, half the memory channels) | 78.25 s | 172.73 s | 452.37 s (96.05 s) | 375.4 / 375.5 / 377.4 GiB |

- **About 20 bytes per value of RAM** (403 GB): 4 for the loaded data, 8 for the presorted columns, 8 for their second copy. It fits in the 512 GB of epyc1 and dell2.
- **epyc is 1.5–1.6× faster than dell** on C4.5 and unpruned CART, as on the 6M sample. Pruned CART is the exception (51.6 against 52.9 s): epyc's presort took 21.2 s there (see below).
- **5.3× the sample's rows:** unpruned CART takes 6.5× (epyc) and 6.3× (dell) the sample's time, C4.5 9.2× and 9.8×; their trees have 4.7× and 5.7× the nodes.

### Filling 375 GiB of a 512 GB node

The presort is where ./tree first touches its 375 GiB, and on the 512 GB nodes it takes longer and spends more time in the kernel (`sys`, all threads together) than on dell1's 1 TB:

| Presort (sys) | pruned CART: first after the data check | C4.5: right after it | unpruned CART: first after the data check |
|---|--:|--:|--:|
| epyc (512 GB) | 21.22 s (972 s) | 5.43 s (191 s) | 7.60 s (326 s) |
| dell512 (512 GB, half the channels) | 12.50 s (343 s) | 13.08 s (236 s) | 12.25 s (280 s) |
| dell (1 TB) | 5.73 s (145 s) | 7.08 s (212 s) | 5.63 s (152 s) |

- **Likely cause: the page cache.** Before each process, `run.py` checks the data files (81 GB `train.f32`, 59 GB `train.csv`) and ./tree reads the CSV, and Linux keeps those files in otherwise free memory. On a 512 GB node that cache and the 375 GiB do not fit side by side, so the kernel drops cache pages while ./tree's threads fault in their memory. On dell1 both fit. The kernel's reclaim counters were not recorded, so this is the most likely reading of the numbers, not a measurement.
- **The cost varies:** 21.2 s in one run on epyc, 7.6 s in the next one that started the same way. A single run per case cannot pin it down.
- **dell512's presort is slower in every case** (12.3–13.1 s against 5.6–7.1 s), about 2× dell1's, more than its half bandwidth alone explains (1.4–1.5× on the 6M sample's presort): the same reclaim, on top of the bandwidth.

## The trees

- **Pruned CART is a single leaf:** at α = 1e-5 every subtree costs more than it saves (with 2.15% contacts a split must fix more than α · n rows per extra leaf), so the pruned tree predicts "no contact" everywhere (training accuracy 97.85%, the majority share). With a fixed α ./tree does not split nodes that misclassify fewer than α · n rows ([docs/ALGORITHMS.md](../../../docs/ALGORITHMS.md)), so the tree it grows before pruning is much smaller than the unpruned one.
- **Unpruned CART:** 255,915 nodes, depth 100 (sample); 1,205,683 nodes, depth 111 (full); training accuracy 100%.
- **C4.5:** 82,811 nodes, depth 224, training accuracy 99.03% (sample); 470,673 nodes, depth 327, 99.19% (full).
