## cuda-hopper-higgs

**cuda-hopper-higgs**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 7f98876f7e2a, CPUs in order of use 16-31,0-15, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 638 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 967 ms | 822 ms–988 ms |  | 2.26 GiB (+ GPU 3.36 GiB) | 1.64 GiB | 2,864,695 | 1,432,348 | 69 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 332 ms | 275 ms–341 ms |  | 1.71 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 357 ms | 350 ms–360 ms |  | 1.71 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 851 ms | 778 ms–868 ms |  | 1.88 GiB (+ GPU 3.36 GiB) | 1.25 GiB | 1,317,065 | 658,533 | 71 | 88.46% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 299 ms | 294 ms–306 ms |  | 1.71 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 63 | 32 | 5 | 66.41% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 314 ms | 302 ms–323 ms |  | 1.71 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 33 | 17 | 5 | 60.54% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 410 ms | 402 ms–413 ms |  | 1.71 GiB (+ GPU 3.36 GiB) | 1.08 GiB | 48,051 | 24,026 | 15 | 73.39% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 419 ms | 365 ms–424 ms |  | 1.71 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 16,029 | 8,015 | 15 | 67.71% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 555 ms | 530 ms–570 ms |  | 1.86 GiB (+ GPU 3.36 GiB) | 1.24 GiB | 426,197 | 213,099 | 20 | 79.50% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 500 ms | 491 ms–503 ms |  | 1.71 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 116,971 | 58,486 | 20 | 71.22% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 749 ms | 735 ms–760 ms |  | 2.25 GiB (+ GPU 3.36 GiB) | 1.63 GiB | 1,324,335 | 662,168 | 25 | 88.93% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 590 ms | 587 ms–602 ms |  | 1.71 GiB (+ GPU 3.33 GiB) | 1.08 GiB | 374,029 | 187,015 | 25 | 76.22% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 861 ms | 844 ms–879 ms |  | 2.14 GiB (+ GPU 3.36 GiB) | 1.52 GiB | 2,158,499 | 1,079,250 | 30 | 95.73% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 680 ms | 661 ms–698 ms |  | 1.71 GiB (+ GPU 3.36 GiB) | 1.08 GiB | 679,355 | 339,678 | 30 | 80.43% |  |
