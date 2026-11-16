## gpuonly-ampere-sorel_4m

**gpuonly-ampere-sorel_4m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree f53347a9d336, CPUs in order of use 0-23,24-47, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree GPU 632 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 33.3 s | 33.1 s–33.4 s |  | 36.44 GiB (+ GPU 76.37 GiB) | 35.82 GiB | 87,729 | 43,865 | 231 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 4.18 s | 4.13 s–4.24 s |  | 36.22 GiB (+ GPU 76.30 GiB) | 35.61 GiB | 1,619 | 810 | 10 | 91.92% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 4.6 s | 4.48 s–4.69 s |  | 39.98 GiB (+ GPU 75.80 GiB) | 39.36 GiB | 299 | 150 | 10 | 81.92% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 79.4 s | 79.1 s–79.5 s |  | 40.04 GiB (+ GPU 76.33 GiB) | 39.42 GiB | 69,637 | 34,819 | 572 | 99.81% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 2.89 s | 2.84 s–2.95 s |  | 36.14 GiB (+ GPU 75.80 GiB) | 35.52 GiB | 63 | 32 | 5 | 85.38% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 3.28 s | 3.15 s–3.36 s |  | 39.97 GiB (+ GPU 75.80 GiB) | 39.36 GiB | 55 | 28 | 5 | 80.14% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 5.78 s | 5.73 s–5.96 s |  | 36.32 GiB (+ GPU 76.37 GiB) | 35.70 GiB | 9,175 | 4,588 | 15 | 95.86% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 5.81 s | 5.73 s–5.89 s |  | 39.99 GiB (+ GPU 75.83 GiB) | 39.37 GiB | 799 | 400 | 15 | 83.14% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 7.49 s | 7.38 s–7.57 s |  | 36.43 GiB (+ GPU 76.37 GiB) | 35.82 GiB | 21,561 | 10,781 | 20 | 97.73% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 7.12 s | 6.99 s–7.21 s |  | 39.99 GiB (+ GPU 75.83 GiB) | 39.38 GiB | 1,577 | 789 | 20 | 85.70% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 9.05 s | 9 s–9.17 s |  | 36.44 GiB (+ GPU 76.37 GiB) | 35.82 GiB | 34,045 | 17,023 | 25 | 98.64% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 8.42 s | 8.32 s–8.5 s |  | 40.01 GiB (+ GPU 76.33 GiB) | 39.39 GiB | 2,701 | 1,351 | 25 | 87.11% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 10.6 s | 10.4 s–10.7 s |  | 36.44 GiB (+ GPU 76.37 GiB) | 35.82 GiB | 43,487 | 21,744 | 30 | 99.08% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 9.78 s | 9.69 s–9.8 s |  | 40.01 GiB (+ GPU 76.33 GiB) | 39.39 GiB | 4,079 | 2,040 | 30 | 88.12% |  |
