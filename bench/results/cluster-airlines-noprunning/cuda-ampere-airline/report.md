## cuda-ampere-airline

**cuda-ampere-airline**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 2ced36833c6c, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 223 MiB

### airline (114,569,017 train / 500,000 test rows, 13 features, 2 classes)

**CART, grown to pure leaves (depth <= 30), no pruning** (`cart_full`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.2 s | 5.2 s–5.21 s |  | 18.81 GiB (+ GPU 23.91 GiB) | 18.59 GiB | 21,600,857 | 10,800,429 | 30 | 89.67% | 77.14% |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 7 s | 6.98 s–7.07 s |  | 25.09 GiB (+ GPU 23.91 GiB) | 24.87 GiB | 19,579,127 | 9,789,564 | 74 | 90.60% | 77.90% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.12 percentage points at most.
