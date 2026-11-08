## cuda-hopper-higgs

**cuda-hopper-higgs**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 95b6d774, CPUs in order of use 16-31,0-15, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 647 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 809 ms | 806 ms–813 ms |  | 4.20 GiB (+ GPU 3.36 GiB) | 3.57 GiB | 2,864,695 | 1,432,348 | 69 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 160 ms | 160 ms–162 ms |  | 3.96 GiB (+ GPU 3.33 GiB) | 3.33 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 172 ms | 167 ms–176 ms |  | 3.99 GiB (+ GPU 3.33 GiB) | 3.36 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 599 ms | 590 ms–604 ms |  | 4.19 GiB (+ GPU 3.36 GiB) | 3.56 GiB | 1,317,065 | 658,533 | 71 | 88.46% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 117 ms | 112 ms–117 ms |  | 3.95 GiB (+ GPU 3.33 GiB) | 3.31 GiB | 63 | 32 | 5 | 66.41% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 124 ms | 122 ms–128 ms |  | 3.98 GiB (+ GPU 3.33 GiB) | 3.35 GiB | 33 | 17 | 5 | 60.54% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 229 ms | 226 ms–234 ms |  | 3.98 GiB (+ GPU 3.36 GiB) | 3.34 GiB | 48,051 | 24,026 | 15 | 73.39% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 229 ms | 228 ms–235 ms |  | 4.00 GiB (+ GPU 3.33 GiB) | 3.36 GiB | 16,029 | 8,015 | 15 | 67.71% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 347 ms | 344 ms–350 ms |  | 4.02 GiB (+ GPU 3.36 GiB) | 3.39 GiB | 426,197 | 213,099 | 20 | 79.50% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 301 ms | 298 ms–302 ms |  | 4.01 GiB (+ GPU 3.36 GiB) | 3.38 GiB | 116,971 | 58,486 | 20 | 71.22% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 515 ms | 510 ms–519 ms |  | 4.08 GiB (+ GPU 3.36 GiB) | 3.45 GiB | 1,324,335 | 662,168 | 25 | 88.93% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 377 ms | 370 ms–379 ms |  | 4.05 GiB (+ GPU 3.36 GiB) | 3.42 GiB | 374,029 | 187,015 | 25 | 76.22% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 672 ms | 670 ms–681 ms |  | 4.16 GiB (+ GPU 3.36 GiB) | 3.53 GiB | 2,158,499 | 1,079,250 | 30 | 95.73% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 452 ms | 443 ms–456 ms |  | 4.10 GiB (+ GPU 3.36 GiB) | 3.47 GiB | 679,355 | 339,678 | 30 | 80.43% |  |
