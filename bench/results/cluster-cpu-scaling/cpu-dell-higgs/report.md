## cpu-dell-higgs

**cpu-dell-higgs**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 1007 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor powersave, turbo on, ./tree ec77727, CPUs in order of use 0-47, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 15.6 s | 15.4 s–15.7 s | 1.00× | 3.57 GiB | 3.56 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.62 s | 8.47 s–9.03 s | 1.00× | 3.64 GiB | 3.64 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 5.61 s | 4.75 s–7.07 s | 1.00× | 3.80 GiB | 3.80 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.8 s | 2.6 s–3.01 s | 1.00× | 4.11 GiB | 4.11 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.04 s | 1.91 s–2.22 s | 1.00× | 4.43 GiB | 4.42 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.62 s | 1.51 s–1.7 s | 1.00× | 4.74 GiB | 4.74 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.37 s | 1.31 s–1.47 s | 1.00× | 5.06 GiB | 5.05 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.21 s | 1.17 s–1.29 s | 1.00× | 5.37 GiB | 5.37 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.07 s | 1.03 s–1.11 s | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1 s | 980 ms–1.03 s | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 935 ms | 897 ms–970 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 883 ms | 847 ms–902 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 826 ms | 810 ms–871 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 812 ms | 763 ms–874 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 20.3 s | 20.2 s–20.6 s | 1.00× | 3.60 GiB | 3.59 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 11.9 s | 11.9 s–12.6 s | 1.00× | 3.68 GiB | 3.67 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.24 s | 6.3 s–9.96 s | 1.00× | 3.83 GiB | 3.83 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.44 s | 3.25 s–3.73 s | 1.00× | 4.15 GiB | 4.14 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.5 s | 2.42 s–2.58 s | 1.00× | 4.46 GiB | 4.46 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.92 s | 1.87 s–1.98 s | 1.00× | 4.78 GiB | 4.77 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.72 s | 1.68 s–1.77 s | 1.00× | 5.09 GiB | 5.08 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.59 s | 1.55 s–1.63 s | 1.00× | 5.40 GiB | 5.40 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.25 s | 1.22 s–1.34 s | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.13 s | 1.1 s–1.16 s | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.08 s | 1.05 s–1.12 s | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.03 s | 1 s–1.08 s | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 988 ms | 967 ms–1.03 s | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 989 ms | 937 ms–1.04 s | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% | 63.73% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
