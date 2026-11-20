## cpu-dell-higgs

**cpu-dell-higgs**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor powersave, turbo on, ./tree 81bf6e6c8979, CPUs in order of use 0-47, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 14.4 s | 12.9 s–17.3 s | 1.00× | 3.49 GiB | 3.48 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 7.58 s | 6.99 s–11.4 s | 1.00× | 3.57 GiB | 3.56 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.4 s | 4.12 s–7.61 s | 1.00× | 3.73 GiB | 3.72 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.34 s | 1.94 s–3.01 s | 1.00× | 4.05 GiB | 4.05 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.61 s | 1.48 s–2.08 s | 1.00× | 4.38 GiB | 4.37 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.28 s | 1.2 s–1.36 s | 1.00× | 4.70 GiB | 4.70 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.1 s | 971 ms–1.19 s | 1.00× | 5.03 GiB | 5.02 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 956 ms | 902 ms–1.08 s | 1.00× | 5.35 GiB | 5.34 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 865 ms | 792 ms–933 ms | 1.00× | 5.59 GiB | 5.59 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 803 ms | 735 ms–889 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 808 ms | 750 ms–932 ms | 1.00× | 5.62 GiB | 5.61 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 777 ms | 682 ms–830 ms | 1.00× | 5.62 GiB | 5.62 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 719 ms | 657 ms–853 ms | 1.00× | 5.64 GiB | 5.63 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 694 ms | 661 ms–810 ms | 1.00× | 5.64 GiB | 5.64 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 12.8 s | 12.5 s–16.9 s | 1.00× | 3.52 GiB | 3.51 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.43 s | 7.59 s–12.5 s | 1.00× | 3.60 GiB | 3.59 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 5.5 s | 4.46 s–7.19 s | 1.00× | 3.76 GiB | 3.76 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.46 s | 2.33 s–3.52 s | 1.00× | 4.09 GiB | 4.08 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.71 s | 1.61 s–2.29 s | 1.00× | 4.41 GiB | 4.41 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.36 s | 1.26 s–1.75 s | 1.00× | 4.74 GiB | 4.73 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.23 s | 1.13 s–1.27 s | 1.00× | 5.06 GiB | 5.06 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.06 s | 1.02 s–1.19 s | 1.00× | 5.38 GiB | 5.37 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 930 ms | 883 ms–1.04 s | 1.00× | 5.62 GiB | 5.62 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 890 ms | 872 ms–1.02 s | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 898 ms | 805 ms–1.01 s | 1.00× | 5.65 GiB | 5.64 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 854 ms | 786 ms–935 ms | 1.00× | 5.65 GiB | 5.65 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 817 ms | 775 ms–956 ms | 1.00× | 5.66 GiB | 5.66 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 840 ms | 817 ms–888 ms | 1.00× | 5.67 GiB | 5.66 GiB | 935 | 468 | 10 | 63.76% |  |
