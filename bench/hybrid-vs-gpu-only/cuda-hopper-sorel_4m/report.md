## cuda-hopper-sorel_4m

**cuda-hopper-sorel_4m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 81bf6e6c8979, CPUs in order of use 16-31,0-15, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 638 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 20 s | 19.9 s–20.1 s |  | 107.14 GiB (+ GPU 75.87 GiB) | 106.51 GiB | 87,729 | 43,865 | 231 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 3.06 s | 3.04 s–3.08 s |  | 107.15 GiB (+ GPU 75.87 GiB) | 106.53 GiB | 1,619 | 810 | 10 | 91.92% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 3.36 s | 3.34 s–3.43 s |  | 110.95 GiB (+ GPU 75.84 GiB) | 110.32 GiB | 299 | 150 | 10 | 81.92% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 52.4 s | 52 s–52.5 s |  | 110.97 GiB (+ GPU 75.84 GiB) | 110.34 GiB | 69,637 | 34,819 | 572 | 99.81% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 2.16 s | 2.14 s–2.18 s |  | 107.10 GiB (+ GPU 75.84 GiB) | 106.48 GiB | 63 | 32 | 5 | 85.38% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 2.48 s | 2.45 s–2.55 s |  | 110.94 GiB (+ GPU 75.84 GiB) | 110.32 GiB | 55 | 28 | 5 | 80.14% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 4.12 s | 4.03 s–4.16 s |  | 107.13 GiB (+ GPU 75.87 GiB) | 106.51 GiB | 9,175 | 4,588 | 15 | 95.86% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 4.21 s | 4.17 s–4.28 s |  | 110.95 GiB (+ GPU 75.84 GiB) | 110.33 GiB | 799 | 400 | 15 | 83.14% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 5.08 s | 5.01 s–5.12 s |  | 107.13 GiB (+ GPU 75.87 GiB) | 106.51 GiB | 21,561 | 10,781 | 20 | 97.73% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 5.06 s | 5.04 s–5.11 s |  | 110.95 GiB (+ GPU 75.84 GiB) | 110.33 GiB | 1,577 | 789 | 20 | 85.70% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 5.8 s | 5.77 s–5.92 s |  | 107.13 GiB (+ GPU 75.87 GiB) | 106.51 GiB | 34,045 | 17,023 | 25 | 98.64% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 5.89 s | 5.85 s–5.94 s |  | 110.95 GiB (+ GPU 75.84 GiB) | 110.33 GiB | 2,701 | 1,351 | 25 | 87.11% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 6.53 s | 6.46 s–6.63 s |  | 107.13 GiB (+ GPU 75.87 GiB) | 106.51 GiB | 43,487 | 21,744 | 30 | 99.08% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 6.75 s | 6.73 s–6.78 s |  | 110.95 GiB (+ GPU 75.84 GiB) | 110.33 GiB | 4,079 | 2,040 | 30 | 88.12% |  |
