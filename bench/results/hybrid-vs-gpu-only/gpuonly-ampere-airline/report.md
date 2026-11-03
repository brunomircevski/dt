## gpuonly-ampere-airline

**gpuonly-ampere-airline**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree fef9ac0, CPUs in order of use 0-23, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 641 MiB

### airline (109,258,555 train / 5,810,462 test rows, 13 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 24.8 s | 24.8 s–24.9 s |  | 9.66 GiB (+ GPU 16.19 GiB) | 9.03 GiB | 44,663,207 | 22,331,604 | 66 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 795 ms | 791 ms–800 ms |  | 6.14 GiB (+ GPU 12.66 GiB) | 5.51 GiB | 2,047 | 1,024 | 10 | 67.10% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 810 ms | 808 ms–810 ms |  | 6.14 GiB (+ GPU 12.66 GiB) | 5.51 GiB | 929 | 465 | 10 | 66.62% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 18.4 s | 18.3 s–18.4 s |  | 8.84 GiB (+ GPU 14.44 GiB) | 8.21 GiB | 18,610,447 | 9,305,224 | 73 | 90.74% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 546 ms | 545 ms–546 ms |  | 6.13 GiB (+ GPU 12.66 GiB) | 5.51 GiB | 63 | 32 | 5 | 60.93% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 551 ms | 550 ms–552 ms |  | 6.13 GiB (+ GPU 12.66 GiB) | 5.51 GiB | 39 | 20 | 5 | 60.79% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 1.08 s | 1.07 s–1.08 s |  | 6.15 GiB (+ GPU 12.72 GiB) | 5.52 GiB | 59,361 | 29,681 | 15 | 71.64% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 1.11 s | 1.1 s–1.11 s |  | 6.15 GiB (+ GPU 12.72 GiB) | 5.53 GiB | 22,971 | 11,486 | 15 | 70.38% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 1.77 s | 1.77 s–1.78 s |  | 6.37 GiB (+ GPU 13.10 GiB) | 5.75 GiB | 994,483 | 497,242 | 20 | 76.43% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 1.67 s | 1.66 s–1.67 s |  | 6.28 GiB (+ GPU 12.91 GiB) | 5.65 GiB | 295,121 | 147,561 | 20 | 73.95% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 4.85 s | 4.83 s–4.85 s |  | 7.21 GiB (+ GPU 14.97 GiB) | 6.59 GiB | 6,763,533 | 3,381,767 | 25 | 82.31% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 3.26 s | 3.25 s–3.27 s |  | 6.65 GiB (+ GPU 13.63 GiB) | 6.02 GiB | 1,729,967 | 864,984 | 25 | 77.94% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 11.9 s | 11.9 s–12 s |  | 8.56 GiB (+ GPU 16.19 GiB) | 7.94 GiB | 20,445,231 | 10,222,616 | 30 | 89.76% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 6.75 s | 6.73 s–6.78 s |  | 7.15 GiB (+ GPU 14.38 GiB) | 6.53 GiB | 5,341,159 | 2,670,580 | 30 | 82.17% |  |
