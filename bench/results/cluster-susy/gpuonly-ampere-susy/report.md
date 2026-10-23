## gpuonly-ampere-susy

**gpuonly-ampere-susy**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 74a011fda9d3, CPUs in order of use 0-23, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 641 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 96 ms | 94 ms–96 ms |  | 964 MiB (+ GPU 1.63 GiB) | 323 MiB | 1,959 | 980 | 10 | 79.16% | 79.01% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 115 ms | 113 ms–115 ms |  | 1.02 GiB (+ GPU 1.63 GiB) | 406 MiB | 743 | 372 | 10 | 78.74% | 78.65% |

**CART, grown to depth 10, pruned at a fixed alpha (cost-complexity)** (`cart_depth10_alpha`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 96 ms | 95 ms–98 ms |  | 964 MiB (+ GPU 1.63 GiB) | 323 MiB | 339 | 170 | 10 | 79.08% | 78.99% |

**C4.5, grown to depth 10, error-based pruning (CF 0.25, subtree raising)** (`c45_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 161 ms | 161 ms–163 ms |  | 1.39 GiB (+ GPU 1.95 GiB) | 786 MiB | 573 | 287 | 10 | 78.74% | 78.64% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
