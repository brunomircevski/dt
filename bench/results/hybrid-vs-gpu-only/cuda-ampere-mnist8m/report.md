## cuda-ampere-mnist8m

**cuda-ampere-mnist8m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 95b6d774, CPUs in order of use 0-23,24-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 17 s | 16.9 s–17 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 301,939 | 150,970 | 115 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.53 s | 5.32 s–5.55 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.05 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.45 s | 5.28 s–5.48 s |  | 71.67 GiB (+ GPU 48.84 GiB) | 71.04 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 22.1 s | 22.1 s–22.1 s |  | 71.70 GiB (+ GPU 48.84 GiB) | 71.08 GiB | 218,325 | 109,163 | 147 | 99.61% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.24 s | 3.24 s–3.29 s |  | 71.66 GiB (+ GPU 48.81 GiB) | 71.03 GiB | 63 | 32 | 5 | 57.28% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.32 s | 3.32 s–3.34 s |  | 71.65 GiB (+ GPU 48.81 GiB) | 71.03 GiB | 61 | 31 | 5 | 44.80% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 7.47 s | 7.46 s–7.47 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.05 GiB | 44,359 | 22,180 | 15 | 92.45% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 7.49 s | 7.42 s–7.55 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.05 GiB | 12,313 | 6,157 | 15 | 82.42% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 8.72 s | 8.67 s–8.85 s |  | 71.69 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 168,241 | 84,121 | 20 | 97.59% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 9.16 s | 9.1 s–9.24 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 42,915 | 21,458 | 20 | 89.16% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 9.86 s | 9.78 s–9.94 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 231,991 | 115,996 | 25 | 98.95% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 10.6 s | 10.5 s–10.7 s |  | 71.68 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 84,917 | 42,459 | 25 | 93.54% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 10.9 s | 10.9 s–11.2 s |  | 71.69 GiB (+ GPU 48.84 GiB) | 71.07 GiB | 257,021 | 128,511 | 30 | 99.42% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 12 s | 11.9 s–12 s |  | 71.69 GiB (+ GPU 48.84 GiB) | 71.06 GiB | 123,289 | 61,645 | 30 | 96.11% |  |
