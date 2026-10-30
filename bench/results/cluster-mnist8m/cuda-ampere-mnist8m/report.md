## cuda-ampere-mnist8m

**cuda-ampere-mnist8m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree ec77727, CPUs in order of use 0-23, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.19 s | 5.17 s–5.26 s |  | 71.69 GiB (+ GPU 49.19 GiB) | 71.06 GiB | 2,047 | 1,024 | 10 | 80.77% | 84.51% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.32 s | 5.31 s–5.37 s |  | 71.67 GiB (+ GPU 48.84 GiB) | 71.04 GiB | 1,453 | 727 | 10 | 70.96% | 74.65% |

Test accuracy is measured on 10,000 rows: its 95% interval is ±0.85 percentage points at most.
