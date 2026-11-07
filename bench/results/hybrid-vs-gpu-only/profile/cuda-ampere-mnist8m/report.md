## cuda-ampere-mnist8m

**cuda-ampere-mnist8m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 95b6d774, CPUs in order of use 0-23,24-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 17 s | 17 s–17.1 s |  | 71.70 GiB (+ GPU 48.84 GiB) | 71.08 GiB | 301,939 | 150,970 | 115 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.55 s | 5.35 s–5.55 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.05 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.51 s | 5.5 s–5.54 s |  | 71.67 GiB (+ GPU 48.84 GiB) | 71.04 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 22.5 s | 22.1 s–22.6 s |  | 71.71 GiB (+ GPU 48.84 GiB) | 71.09 GiB | 218,325 | 109,163 | 147 | 99.61% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.23 s | 3.13 s–3.25 s |  | 71.66 GiB (+ GPU 48.81 GiB) | 71.03 GiB | 63 | 32 | 5 | 57.28% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.37 s | 3.36 s–3.4 s |  | 71.66 GiB (+ GPU 48.81 GiB) | 71.03 GiB | 61 | 31 | 5 | 44.80% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 7.52 s | 7.52 s–7.62 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 44,359 | 22,180 | 15 | 92.45% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 7.5 s | 7.48 s–7.52 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 12,313 | 6,157 | 15 | 82.42% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 8.81 s | 8.74 s–8.84 s |  | 71.69 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 168,241 | 84,121 | 20 | 97.59% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 9.19 s | 9.16 s–9.21 s |  | 71.69 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 42,915 | 21,458 | 20 | 89.16% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 9.86 s | 9.83 s–9.93 s |  | 71.69 GiB (+ GPU 48.84 GiB) | 71.07 GiB | 231,991 | 115,996 | 25 | 98.95% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 10.6 s | 10.6 s–10.6 s |  | 71.69 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 84,917 | 42,459 | 25 | 93.54% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 11 s | 11 s–11 s |  | 71.70 GiB (+ GPU 48.84 GiB) | 71.08 GiB | 257,021 | 128,511 | 30 | 99.42% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 12 s | 12 s–12 s |  | 71.69 GiB (+ GPU 48.84 GiB) | 71.07 GiB | 123,289 | 61,645 | 30 | 96.11% |  |
