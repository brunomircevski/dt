## cuda-ampere-ecbdl_15m

**cuda-ampere-ecbdl_15m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 74a011fda9d3, CPUs in order of use 0-23, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### ecbdl_15m (15,000,000 train / 2,897,917 test rows, 631 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.85 s | 5.82 s–5.85 s |  | 106.52 GiB (+ GPU 73.30 GiB) | 105.89 GiB | 1,863 | 932 | 10 | 97.92% | 98.31% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 6.03 s | 5.94 s–6.04 s |  | 106.50 GiB (+ GPU 73.30 GiB) | 105.88 GiB | 725 | 363 | 10 | 97.92% | 98.32% |

**CART, grown to depth 10, pruned at a fixed alpha (cost-complexity)** (`cart_depth10_alpha`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.8 s | 5.77 s–5.81 s |  | 36.16 GiB (+ GPU 73.26 GiB) | 35.53 GiB | 1 | 1 | 0 | 97.91% | 98.32% |

**C4.5, grown to depth 10, error-based pruning (CF 0.25, subtree raising)** (`c45_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 6.09 s | 6.08 s–6.18 s |  | 106.79 GiB (+ GPU 73.32 GiB) | 106.16 GiB | 161 | 81 | 10 | 97.91% | 98.32% |

Test accuracy is measured on 2,897,917 rows: its 95% interval is ±0.01 percentage points at most.
