## cuda-ampere-sorel_4m

**cuda-ampere-sorel_4m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 95b6d774, CPUs in order of use 0-23,24-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 35.2 s | 35.2 s–35.3 s |  | 107.14 GiB (+ GPU 75.83 GiB) | 106.52 GiB | 87,729 | 43,865 | 231 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 4.93 s | 4.89 s–5 s |  | 107.14 GiB (+ GPU 75.83 GiB) | 106.51 GiB | 1,619 | 810 | 10 | 91.92% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 5.26 s | 5.22 s–5.32 s |  | 110.96 GiB (+ GPU 75.80 GiB) | 110.34 GiB | 299 | 150 | 10 | 81.92% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 95 s | 94.9 s–95 s |  | 110.98 GiB (+ GPU 75.83 GiB) | 110.35 GiB | 69,637 | 34,819 | 572 | 99.81% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.17 s | 3.14 s–3.2 s |  | 107.13 GiB (+ GPU 75.83 GiB) | 106.50 GiB | 63 | 32 | 5 | 85.38% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.74 s | 3.64 s–3.78 s |  | 110.96 GiB (+ GPU 75.80 GiB) | 110.34 GiB | 55 | 28 | 5 | 80.14% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 6.69 s | 6.6 s–6.72 s |  | 107.14 GiB (+ GPU 75.83 GiB) | 106.51 GiB | 9,175 | 4,588 | 15 | 95.86% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 6.78 s | 6.77 s–6.82 s |  | 110.96 GiB (+ GPU 75.83 GiB) | 110.34 GiB | 799 | 400 | 15 | 83.14% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 8.05 s | 8 s–8.05 s |  | 107.14 GiB (+ GPU 75.83 GiB) | 106.51 GiB | 21,561 | 10,781 | 20 | 97.73% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 8.4 s | 8.31 s–8.44 s |  | 110.96 GiB (+ GPU 75.83 GiB) | 110.34 GiB | 1,577 | 789 | 20 | 85.70% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 9.4 s | 9.33 s–9.46 s |  | 107.14 GiB (+ GPU 75.83 GiB) | 106.51 GiB | 34,045 | 17,023 | 25 | 98.64% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 9.84 s | 9.82 s–10 s |  | 110.97 GiB (+ GPU 75.83 GiB) | 110.34 GiB | 2,701 | 1,351 | 25 | 87.11% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 10.8 s | 10.8 s–10.9 s |  | 107.14 GiB (+ GPU 75.83 GiB) | 106.52 GiB | 43,487 | 21,744 | 30 | 99.08% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 11.4 s | 11.4 s–11.5 s |  | 110.97 GiB (+ GPU 75.83 GiB) | 110.34 GiB | 4,079 | 2,040 | 30 | 88.12% |  |
