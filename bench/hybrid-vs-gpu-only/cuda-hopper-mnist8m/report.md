## cuda-hopper-mnist8m

**cuda-hopper-mnist8m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 7f98876f7e2a, CPUs in order of use 16-31,0-15, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 638 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 8.1 s | 8.05 s–8.2 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 301,939 | 150,970 | 115 | 100.00% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 2.37 s | 2.35 s–2.41 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 2.36 s | 2.34 s–2.37 s |  | 24.26 GiB (+ GPU 48.85 GiB) | 23.64 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 9.55 s | 9.53 s–9.63 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 218,325 | 109,163 | 147 | 99.61% |  |

**CART, grown to depth 5, no pruning** (`cart_depth5`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 1.54 s | 1.52 s–1.56 s |  | 24.26 GiB (+ GPU 48.85 GiB) | 23.64 GiB | 63 | 32 | 5 | 57.28% |  |

**C4.5, grown to depth 5, unpruned (only C4.5's collapse of useless splits)** (`c45_depth5_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 1.58 s | 1.57 s–1.59 s |  | 24.26 GiB (+ GPU 48.85 GiB) | 23.64 GiB | 61 | 31 | 5 | 44.80% |  |

**CART, grown to depth 15, no pruning** (`cart_depth15`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 3.75 s | 3.73 s–3.79 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 44,359 | 22,180 | 15 | 92.45% |  |

**C4.5, grown to depth 15, unpruned (only C4.5's collapse of useless splits)** (`c45_depth15_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 3.27 s | 3.26 s–3.3 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 12,313 | 6,157 | 15 | 82.42% |  |

**CART, grown to depth 20, no pruning** (`cart_depth20`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 4.97 s | 4.93 s–5.01 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 168,241 | 84,121 | 20 | 97.59% |  |

**C4.5, grown to depth 20, unpruned (only C4.5's collapse of useless splits)** (`c45_depth20_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 4.1 s | 4.07 s–4.12 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 42,915 | 21,458 | 20 | 89.16% |  |

**CART, grown to depth 25, no pruning** (`cart_depth25`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 5.87 s | 5.84 s–5.9 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 231,991 | 115,996 | 25 | 98.95% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 4.93 s | 4.9 s–4.99 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 84,917 | 42,459 | 25 | 93.54% |  |

**CART, grown to depth 30, no pruning** (`cart_depth30`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 6.47 s | 6.45 s–6.49 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 257,021 | 128,511 | 30 | 99.42% |  |

**C4.5, grown to depth 30, unpruned (only C4.5's collapse of useless splits)** (`c45_depth30_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 5.53 s | 5.49 s–5.6 s |  | 24.26 GiB (+ GPU 48.88 GiB) | 23.64 GiB | 123,289 | 61,645 | 30 | 96.11% |  |
