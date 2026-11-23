## cuda-ampere-higgs

**cuda-ampere-higgs**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-43-generic #43~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Sat Oct 31 06:52:47 UTC 2 x86_64, governor ondemand, turbo on, ./tree 7f98876f7e2a, CPUs in order of use 0-23,24-47, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 633 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 945 ms | 913 ms–967 ms |  | 1.72 GiB (+ GPU 3.33 GiB) | 1.11 GiB | 2,864,695 | 1,432,348 | 69 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 382 ms | 366 ms–396 ms |  | 1.70 GiB (+ GPU 3.30 GiB) | 1.08 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 395 ms | 380 ms–410 ms |  | 1.70 GiB (+ GPU 3.30 GiB) | 1.08 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 876 ms | 846 ms–967 ms |  | 1.70 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 1,317,065 | 658,533 | 71 | 88.46% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 334 ms | 325 ms–351 ms |  | 1.70 GiB (+ GPU 3.30 GiB) | 1.08 GiB | 63 | 32 | 5 | 66.41% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 345 ms | 333 ms–351 ms |  | 1.70 GiB (+ GPU 3.30 GiB) | 1.08 GiB | 33 | 17 | 5 | 60.54% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 477 ms | 471 ms–493 ms |  | 1.70 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 48,051 | 24,026 | 15 | 73.39% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 479 ms | 474 ms–493 ms |  | 1.70 GiB (+ GPU 3.30 GiB) | 1.08 GiB | 16,029 | 8,015 | 15 | 67.71% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 673 ms | 640 ms–683 ms |  | 1.70 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 426,197 | 213,099 | 20 | 79.50% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 579 ms | 566 ms–590 ms |  | 1.70 GiB (+ GPU 3.30 GiB) | 1.08 GiB | 116,971 | 58,486 | 20 | 71.22% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 737 ms | 727 ms–823 ms |  | 1.70 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 1,324,335 | 662,168 | 25 | 88.93% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 673 ms | 653 ms–717 ms |  | 1.70 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 374,029 | 187,015 | 25 | 76.22% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 835 ms | 817 ms–859 ms |  | 1.70 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 2,158,499 | 1,079,250 | 30 | 95.73% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 731 ms | 719 ms–766 ms |  | 1.70 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 679,355 | 339,678 | 30 | 80.43% |  |
