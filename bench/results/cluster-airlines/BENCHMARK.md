# Airline on the mordor3 cluster: ./tree CUDA on two GPU servers, parallel on two CPU servers

**Status:** run on 2026-10-14, 3 runs per case. The ./tree source is that of commit 16cc93b, with the defaults tuned for these servers ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)). Later commits changed only `bench/`.
- **CPU nodes:** epyc1 and dell1 ran at 17:56 (commit 7cb9fb9), each right after its HIGGS run ([cluster-higgs](../cluster-higgs/BENCHMARK.md)).
- **GPU nodes:** hopper and ampere ran at 18:08 (commit 67dc294), CUDA only.
- **Each node ran alone** and was used whole.
- **Same trees:** every case built the same tree on every node and backend.

How the runs are made, and the dataset: [bench/cluster/README.md](../../cluster/README.md).
- **GPU servers run only CUDA, CPU servers only parallel.**
- **CART has no depth limit** (`cart_alpha_nodepth`), like C4.5. The pruned tree is depth 25, so a cap at 30 would make no difference.
- **Accuracy is ./tree's training accuracy** (`run.py --train-accuracy-only`).

The dataset is gbm-bench's airline: US flights 1987–2008, 114,569,017 training rows, 13 integer features, label "arrival delayed". It has 11× HIGGS's rows. It exists only on the cluster: the CUDA backend needs 23.6 GB of GPU memory for it (two copies of the presorted columns).

| Run | Node | CPU | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|
| `cuda-hopper-airline` | hopper | 2× Xeon Gold 6444Y, 32 cores | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-airline` | ampere | 2× EPYC 7402, 48 cores | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-airline` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | – | parallel (48 threads) |
| `cpu-dell-airline` | dell1 | 2× Xeon Gold 6542Y, 48 cores | – | parallel (48 threads) |

## Results

![airline on four mordor3 nodes](chart.png)

Each bar is ./tree's training time (`train total`: GPU setup, presort, build and pruning, not reading the 6 GB CSV). Its lighter part is the pruning.

Training time, median of 3 runs (min–max):

| | CART | C4.5 | of which C4.5 pruning |
|---|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | **2.31 s** (2.27–2.34) | 15.83 s (14.85–16.10) | 7.55 s |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 3.37 s (3.37–3.41) | **13.19 s** (13.11–13.28) | 6.60 s |
| Parallel · epyc (EPYC 9255, 48 threads) | 7.34 s (7.32–7.38) | 13.96 s (13.86–14.06) | 2.87 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 8.36 s (8.35–8.49) | 16.60 s (16.43–16.61) | 3.90 s |

The trees:
- **CART:** 3,013 nodes, depth 25, training accuracy 71.54%. It grows 0.1 million nodes before pruning, instead of the 21.6 million it grew before the tuning.
- **C4.5:** 8,548,725 nodes (19.6 million before pruning), depth 72, training accuracy 87.42%.
- **Accuracy:** `ActualElapsedTime` partly gives the delay away, so the accuracies say little. The dataset is here to compare speed, as in gbm-bench.

### Against the code before tuning

The tuning compared both versions on hopper and epyc1, in one job each, interleaved, 3 runs ([docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md)):

| Median of 3 | CART before → after | C4.5 before → after |
|---|--:|--:|
| CUDA · hopper | 11.64 s → 2.32 s (5.0×) | 26.03 s → 14.53 s (1.8×) |
| Parallel · epyc | 15.98 s → 7.34 s (2.2×) | 24.67 s → 13.93 s (1.8×) |

## Observations

- **CART:** CUDA wins clearly. The H100 takes 2.31 s, 3.2× the fastest CPU node (epyc, 7.34 s); the A100 takes 3.37 s. With 115M rows the GPU does most of the work. Its setup (uploading 6 GB: 0.35 s on the H100, 0.51 s on the A100) and the per-level round trips matter less than on HIGGS.
- **C4.5:** the GPUs only tie with epyc's 48 cores (A100 13.2 s, epyc 14.0 s, H100 15.8 s).
  - **Pruning is half of a CUDA run** (6.6–7.6 s). It runs on the host CPU, as do the small subtrees of the 19.6-million-node tree. epyc prunes in 2.9 s.
  - **Growing alone,** without the pruning, the GPUs take 6.6 s (A100) and 8.3 s (H100) against epyc's 11.1 s.
  - **Hopper against ampere:** hopper's 32 cores lose to ampere's 48 in pruning (7.55 against 6.60 s) and in the small subtrees. Hopper's CPU timings also drift during the day: at 16:04 the same binary took 14.19 s there.
  - **What would help next:** C4.5's pruning is now the place to gain on the GPU nodes, through more parallelism or work on the GPU.
- **CPU nodes:** dell takes 14–19% longer than epyc (CART 8.36 against 7.34 s, C4.5 16.60 against 13.96 s).
- **Memory:** the same on every node of a kind.
  - Parallel: 29.0 GiB (CART) and 38.1 GiB (C4.5) peak RSS.
  - CUDA: 17.2 / 26.3–26.5 GiB RSS plus 23.9 GiB on the GPU, CUDA context included.
- **Wall time:** with only ./tree's own training accuracy, a whole CPU-node run (3 runs per case and the data check) took 1¼ minutes. The earlier runs predicted the training and test rows in Python and took about 1.5 minutes per C4.5 run alone.
