## cuda-ampere-susy

**cuda-ampere-susy**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree ec77727, CPUs in order of use 0-23, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 90 ms | 87 ms–97 ms |  | 1.55 GiB (+ GPU 1.63 GiB) | 945 MiB | 1,959 | 980 | 10 | 79.16% | 79.01% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 119 ms | 116 ms–124 ms |  | 1.64 GiB (+ GPU 1.63 GiB) | 1.01 GiB | 743 | 372 | 10 | 78.74% | 78.65% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
