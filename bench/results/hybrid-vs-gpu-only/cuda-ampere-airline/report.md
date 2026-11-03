## cuda-ampere-airline

**cuda-ampere-airline**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree fef9ac0, CPUs in order of use 0-23, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### airline (109,258,555 train / 5,810,462 test rows, 13 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 4.49 s | 4.49 s–4.49 s |  | 20.27 GiB (+ GPU 12.79 GiB) | 19.65 GiB | 44,663,207 | 22,331,604 | 66 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 802 ms | 800 ms–805 ms |  | 16.72 GiB (+ GPU 12.66 GiB) | 16.09 GiB | 2,047 | 1,024 | 10 | 67.10% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 815 ms | 814 ms–816 ms |  | 16.81 GiB (+ GPU 12.66 GiB) | 16.18 GiB | 929 | 465 | 10 | 66.62% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 4.15 s | 4.14 s–4.17 s |  | 18.95 GiB (+ GPU 12.72 GiB) | 18.33 GiB | 18,610,447 | 9,305,224 | 73 | 90.74% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 544 ms | 543 ms–554 ms |  | 16.72 GiB (+ GPU 12.66 GiB) | 16.09 GiB | 63 | 32 | 5 | 60.93% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 562 ms | 561 ms–564 ms |  | 16.72 GiB (+ GPU 12.66 GiB) | 16.09 GiB | 39 | 20 | 5 | 60.79% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 1.08 s | 1.08 s–1.09 s |  | 16.84 GiB (+ GPU 12.66 GiB) | 16.21 GiB | 59,361 | 29,681 | 15 | 71.64% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 1.11 s | 1.11 s–1.11 s |  | 16.84 GiB (+ GPU 12.66 GiB) | 16.22 GiB | 22,971 | 11,486 | 15 | 70.38% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 1.56 s | 1.56 s–1.57 s |  | 16.93 GiB (+ GPU 12.75 GiB) | 16.30 GiB | 994,483 | 497,242 | 20 | 76.43% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 1.49 s | 1.49 s–1.49 s |  | 16.89 GiB (+ GPU 12.69 GiB) | 16.27 GiB | 295,121 | 147,561 | 20 | 73.95% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 2.14 s | 2.13 s–2.15 s |  | 17.40 GiB (+ GPU 12.79 GiB) | 16.78 GiB | 6,763,533 | 3,381,767 | 25 | 82.31% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 2 s | 1.99 s–2.04 s |  | 17.15 GiB (+ GPU 12.72 GiB) | 16.52 GiB | 1,729,967 | 864,984 | 25 | 77.94% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.22 s | 3.21 s–3.28 s |  | 18.49 GiB (+ GPU 12.79 GiB) | 17.86 GiB | 20,445,231 | 10,222,616 | 30 | 89.76% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 2.51 s | 2.51 s–2.52 s |  | 17.78 GiB (+ GPU 12.72 GiB) | 17.16 GiB | 5,341,159 | 2,670,580 | 30 | 82.17% |  |
