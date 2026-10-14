# Airline on the mordor3 cluster: ./tree parallel and CUDA on five servers

**Status:** run on 2026-10-14 from 16:04 to 16:30, 3 runs per case, at commit 16cc93b with the tuned defaults. machine.json records 5f4f955, 971ad14 or 36c6b7c: 16cc93b plus changes to `bench/cluster/` only.
- **Nodes:** each node ran alone, used whole, right after its HIGGS run ([cluster-higgs](../cluster-higgs/BENCHMARK.md)). The runs started within three minutes of each other.
- **Same trees:** every case built the same tree on every node and backend.

How the runs are made, and the dataset: [bench/cluster/README.md](../../cluster/README.md).

The dataset is gbm-bench's airline: US flights 1987–2008, 114,569,017 training rows and 500,000 test rows, 13 integer features, label "arrival delayed". It has 11× HIGGS's rows. It exists only on the cluster: the CUDA backend needs 24 GiB of GPU memory for it.

| Run | Node | CPU | GPU (GPU 0 used) | Backends |
|---|---|---|---|---|
| `cuda-hopper-airline` | hopper | 2× Xeon Gold 6444Y, 32 cores | H100 PCIe 80 GB | parallel (32 threads), CUDA |
| `cuda-ampere-airline` | ampere | 2× EPYC 7402, 48 cores | A100 80 GB PCIe | parallel (48 threads), CUDA |
| `cpu-turing-airline` | turing | Threadripper 1950X, 16 cores, SMT on | (RTX 2080 Ti 11 GB: too small) | parallel (32 threads) |
| `cpu-epyc-airline` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | – | parallel (48 threads) |
| `cpu-dell-airline` | dell1 | 2× Xeon Gold 6542Y, 48 cores | – | parallel (48 threads) |

## Results

![airline on five mordor3 nodes](chart.png)

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the 6 GB CSV). Its lighter part is the pruning.

Training time, median of 3 runs (min–max), and how much of it is pruning:

| | CART | C4.5 | of which C4.5 pruning |
|---|--:|--:|--:|
| Parallel · hopper (Xeon 6444Y, 32 threads) | 9.42 s (9.38–9.83) | 21.17 s (21.06–21.76) | 6.72 s |
| Parallel · ampere (EPYC 7402, 48 threads) | 16.44 s (16.44–17.44) | 32.97 s (32.84–33.55) | 6.98 s |
| Parallel · turing (Threadripper 1950X, 32 threads) | 38.08 s (37.90–38.14) | 64.07 s (63.91–64.17) | 8.18 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 7.37 s (7.32–7.43) | 14.01 s (13.85–14.15) | 2.97 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 8.51 s (8.29–8.92) | 16.49 s (16.16–16.72) | 3.89 s |
| CUDA · hopper (H100 PCIe + 32 threads) | **2.25 s** (2.25–2.29) | 14.19 s (14.16–14.35) | 7.52 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 3.35 s (3.32–3.36) | **13.41 s** (13.31–13.52) | 6.83 s |

The trees:
- **CART:** 3,013 nodes, depth 25, test accuracy 71.59%. It grows 0.1 million nodes before pruning, instead of the 21.6 million it grew before the tuning.
- **C4.5:** 8,548,725 nodes (19.6 million before pruning), depth 72, test accuracy 79.70%.
- **Accuracy:** `ActualElapsedTime` gives the delay partly away, so the accuracies say little. The dataset is here to compare speed, as in gbm-bench.

### Against the code before tuning

The benchmark ran only with the tuned code. The tuning compared both versions on hopper and epyc1, in one job each, interleaved, 3 runs ([cluster-tuning](../cluster-tuning/README.md), round 14):

| Median of 3 | CART before → after | C4.5 before → after |
|---|--:|--:|
| Parallel · epyc | 15.98 s → 7.34 s (2.2×) | 24.67 s → 13.93 s (1.8×) |
| CUDA · hopper | 11.64 s → 2.32 s (5.0×) | 26.03 s → 14.53 s (1.8×) |

## Observations

- **CART:** CUDA wins clearly. The H100 takes 2.25 s, 4.2× hopper's own CPU and 3.3× the fastest CPU node (epyc, 7.37 s); the A100 takes 3.35 s. With 115M rows the GPU does most of the work. Its setup (uploading 6 GB, 0.33 s on the H100) and the per-level round trips matter less than on HIGGS.
- **C4.5:** the GPUs only tie with epyc's 48 cores (H100 14.2 s, A100 13.4 s, epyc 14.0 s). More than half of a CUDA run is now pruning (6.8–7.5 s), and the pruning runs on the host CPU. The 19.6-million-node tree's small subtrees also grow on the host.
  - **Why the GPU nodes lose there:** epyc prunes in 3.0 s, while hopper's and ampere's slower CPUs need 6.7–7.0 s with either backend.
  - **Growing alone:** without the pruning, the GPUs grow the C4.5 tree in 6.6–6.7 s against epyc's 11.0 s.
  - **What would help next:** C4.5's pruning is now the place to gain on the GPU nodes, as more parallelism or work on the GPU.
- **CPU nodes:** epyc is the fastest: CART 7.4 s, C4.5 14.0 s. Dell is 15–18% behind, hopper's 32 cores 28–51%. Ampere's Zen 2 needs 2.2–2.4× epyc's time, and turing's 16 Zen 1 cores 4.6–5.2×.
- **Memory:** about the same on every node, apart from small measurement noise.
  - Parallel: 29 GiB (CART) and 38 GiB (C4.5) peak RSS.
  - CUDA: 17 / 26 GiB RSS plus 23.9 GiB on the GPU, CUDA context included. Turing's 11 GB GPU cannot hold that, so turing ran on the CPU only.
- **Wall time:** a parallel C4.5 run on epyc took about 1.5 minutes. ./tree's process took 17 s of it, the CSV read included; the rest is run.py's untimed evaluation of the 8.5-million-node tree in Python.
