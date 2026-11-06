## cuda-hopper-airline

**cuda-hopper-airline**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree c992b84, CPUs in order of use 16-31, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 648 MiB

### airline (109,258,555 train / 5,810,462 test rows, 13 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 4.39 s | 4.36 s–4.44 s |  | 20.37 GiB (+ GPU 12.88 GiB) | 19.74 GiB | 44,663,207 | 22,331,604 | 66 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 497 ms | 493 ms–498 ms |  | 16.73 GiB (+ GPU 12.69 GiB) | 16.10 GiB | 2,047 | 1,024 | 10 | 67.10% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 508 ms | 508 ms–515 ms |  | 16.80 GiB (+ GPU 12.69 GiB) | 16.17 GiB | 929 | 465 | 10 | 66.62% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.91 s | 3.88 s–3.98 s |  | 19.78 GiB (+ GPU 12.82 GiB) | 19.15 GiB | 18,610,447 | 9,305,224 | 73 | 90.74% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 324 ms | 318 ms–325 ms |  | 16.72 GiB (+ GPU 12.69 GiB) | 16.09 GiB | 63 | 32 | 5 | 60.93% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 329 ms | 328 ms–332 ms |  | 16.72 GiB (+ GPU 12.69 GiB) | 16.09 GiB | 39 | 20 | 5 | 60.79% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 692 ms | 691 ms–698 ms |  | 16.85 GiB (+ GPU 12.69 GiB) | 16.21 GiB | 59,361 | 29,681 | 15 | 71.64% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 722 ms | 719 ms–726 ms |  | 16.85 GiB (+ GPU 12.69 GiB) | 16.21 GiB | 22,971 | 11,486 | 15 | 70.38% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 1.18 s | 1.17 s–1.19 s |  | 16.97 GiB (+ GPU 12.82 GiB) | 16.34 GiB | 994,483 | 497,242 | 20 | 76.43% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 1.07 s | 1.06 s–1.07 s |  | 16.90 GiB (+ GPU 12.78 GiB) | 16.27 GiB | 295,121 | 147,561 | 20 | 73.95% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 1.95 s | 1.92 s–2 s |  | 17.47 GiB (+ GPU 12.88 GiB) | 16.84 GiB | 6,763,533 | 3,381,767 | 25 | 82.31% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 1.66 s | 1.64 s–1.69 s |  | 17.17 GiB (+ GPU 12.82 GiB) | 16.54 GiB | 1,729,967 | 864,984 | 25 | 77.94% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 2.86 s | 2.84 s–2.92 s |  | 18.62 GiB (+ GPU 12.88 GiB) | 17.99 GiB | 20,445,231 | 10,222,616 | 30 | 89.76% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 2.32 s | 2.27 s–2.33 s |  | 17.81 GiB (+ GPU 12.82 GiB) | 17.17 GiB | 5,341,159 | 2,670,580 | 30 | 82.17% |  |
