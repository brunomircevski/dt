## gpuonly-hopper-airline

**gpuonly-hopper-airline**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 95b6d774, CPUs in order of use 16-31,0-15, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 649 MiB

### airline (109,258,555 train / 5,810,462 test rows, 13 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 21.8 s | 21.7 s–21.8 s |  | 9.70 GiB (+ GPU 16.19 GiB) | 9.06 GiB | 44,663,207 | 22,331,604 | 66 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 492 ms | 492 ms–496 ms |  | 6.14 GiB (+ GPU 12.69 GiB) | 5.51 GiB | 2,047 | 1,024 | 10 | 67.10% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 507 ms | 504 ms–508 ms |  | 6.14 GiB (+ GPU 12.69 GiB) | 5.51 GiB | 929 | 465 | 10 | 66.62% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 15.6 s | 15.5 s–15.8 s |  | 8.91 GiB (+ GPU 14.44 GiB) | 8.27 GiB | 18,610,447 | 9,305,224 | 73 | 90.74% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 322 ms | 319 ms–322 ms |  | 6.14 GiB (+ GPU 12.69 GiB) | 5.51 GiB | 63 | 32 | 5 | 60.93% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 324 ms | 323 ms–326 ms |  | 6.14 GiB (+ GPU 12.69 GiB) | 5.51 GiB | 39 | 20 | 5 | 60.79% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 741 ms | 738 ms–745 ms |  | 6.16 GiB (+ GPU 12.75 GiB) | 5.53 GiB | 59,361 | 29,681 | 15 | 71.64% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 752 ms | 726 ms–757 ms |  | 6.16 GiB (+ GPU 12.75 GiB) | 5.52 GiB | 22,971 | 11,486 | 15 | 70.38% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 1.38 s | 1.38 s–1.39 s |  | 6.39 GiB (+ GPU 13.13 GiB) | 5.76 GiB | 994,483 | 497,242 | 20 | 76.43% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 1.21 s | 1.21 s–1.25 s |  | 6.27 GiB (+ GPU 12.91 GiB) | 5.64 GiB | 295,121 | 147,561 | 20 | 73.95% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 4.61 s | 4.53 s–4.7 s |  | 7.22 GiB (+ GPU 15.07 GiB) | 6.59 GiB | 6,763,533 | 3,381,767 | 25 | 82.31% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 2.83 s | 2.77 s–2.85 s |  | 6.65 GiB (+ GPU 13.66 GiB) | 6.02 GiB | 1,729,967 | 864,984 | 25 | 77.94% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 10.8 s | 10.7 s–11 s |  | 8.57 GiB (+ GPU 16.19 GiB) | 7.94 GiB | 20,445,231 | 10,222,616 | 30 | 89.76% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 6.02 s | 5.98 s–6.05 s |  | 7.15 GiB (+ GPU 14.41 GiB) | 6.52 GiB | 5,341,159 | 2,670,580 | 30 | 82.17% |  |
