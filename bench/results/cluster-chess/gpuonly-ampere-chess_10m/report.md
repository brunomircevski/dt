## gpuonly-ampere-chess_10m

**gpuonly-ampere-chess_10m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree ec77727, CPUs in order of use 0-23, 30 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 641 MiB

### chess_10m (10,000,000 train / 4,500 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 46 ms | 43 ms–50 ms |  | 741 MiB (+ GPU 841 MiB) | 100 MiB | 17 | 9 | 4 | 100.00% | 99.98% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 48 ms | 44 ms–55 ms |  | 742 MiB (+ GPU 873 MiB) | 101 MiB | 17 | 9 | 4 | 100.00% | 100.00% |

Test accuracy is measured on 4,500 rows: its 95% interval is ±0.04 percentage points at most.
