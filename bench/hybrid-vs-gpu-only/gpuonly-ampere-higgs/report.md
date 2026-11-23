## gpuonly-ampere-higgs

**gpuonly-ampere-higgs**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-43-generic #43~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Sat Oct 31 06:52:47 UTC 2 x86_64, governor ondemand, turbo on, ./tree 7f98876f7e2a, CPUs in order of use 0-23,24-47, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree GPU 378 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 1.64 s | 1.61 s–1.71 s |  | 2.07 GiB (+ GPU 3.67 GiB) | 1.70 GiB | 2,864,695 | 1,432,348 | 69 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 244 ms | 239 ms–251 ms |  | 1.51 GiB (+ GPU 2.80 GiB) | 1.14 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 266 ms | 263 ms–278 ms |  | 1.52 GiB (+ GPU 2.83 GiB) | 1.15 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 1.57 s | 1.54 s–1.61 s |  | 1.86 GiB (+ GPU 3.20 GiB) | 1.49 GiB | 1,317,065 | 658,533 | 71 | 88.46% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 189 ms | 185 ms–207 ms |  | 1.51 GiB (+ GPU 2.80 GiB) | 1.14 GiB | 63 | 32 | 5 | 66.41% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 206 ms | 203 ms–216 ms |  | 1.52 GiB (+ GPU 2.84 GiB) | 1.15 GiB | 33 | 17 | 5 | 60.54% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 351 ms | 345 ms–365 ms |  | 1.53 GiB (+ GPU 2.86 GiB) | 1.16 GiB | 48,051 | 24,026 | 15 | 73.39% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 363 ms | 353 ms–369 ms |  | 1.54 GiB (+ GPU 2.83 GiB) | 1.17 GiB | 16,029 | 8,015 | 15 | 67.71% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 598 ms | 588 ms–602 ms |  | 1.76 GiB (+ GPU 3.36 GiB) | 1.39 GiB | 426,197 | 213,099 | 20 | 79.50% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 535 ms | 524 ms–544 ms |  | 1.62 GiB (+ GPU 2.98 GiB) | 1.25 GiB | 116,971 | 58,486 | 20 | 71.22% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 962 ms | 937 ms–971 ms |  | 1.96 GiB (+ GPU 3.67 GiB) | 1.60 GiB | 1,324,335 | 662,168 | 25 | 88.93% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 766 ms | 754 ms–789 ms |  | 1.73 GiB (+ GPU 3.19 GiB) | 1.36 GiB | 374,029 | 187,015 | 25 | 76.22% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 1.29 s | 1.26 s–1.31 s |  | 2.02 GiB (+ GPU 3.67 GiB) | 1.65 GiB | 2,158,499 | 1,079,250 | 30 | 95.73% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 995 ms | 987 ms–1.01 s |  | 1.78 GiB (+ GPU 3.20 GiB) | 1.41 GiB | 679,355 | 339,678 | 30 | 80.43% |  |
