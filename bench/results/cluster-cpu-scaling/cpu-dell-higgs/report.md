## cpu-dell-higgs

**cpu-dell-higgs**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 1007 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor powersave, turbo on, ./tree f53347a9d336, CPUs in order of use 0-47, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 12.7 s | 12.5 s–12.8 s | 1.00× | 3.49 GiB | 3.48 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 6.63 s | 6.5 s–6.96 s | 1.00× | 3.57 GiB | 3.56 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.86 s | 3.71 s–5.77 s | 1.00× | 3.73 GiB | 3.72 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.16 s | 2.14 s–2.49 s | 1.00× | 4.05 GiB | 4.05 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.45 s | 1.41 s–1.59 s | 1.00× | 4.38 GiB | 4.37 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.15 s | 1.11 s–1.26 s | 1.00× | 4.70 GiB | 4.70 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 981 ms | 971 ms–1.04 s | 1.00× | 5.03 GiB | 5.02 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 852 ms | 820 ms–872 ms | 1.00× | 5.35 GiB | 5.34 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 746 ms | 721 ms–760 ms | 1.00× | 5.59 GiB | 5.58 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 675 ms | 662 ms–710 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 636 ms | 617 ms–644 ms | 1.00× | 5.62 GiB | 5.61 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 596 ms | 579 ms–625 ms | 1.00× | 5.62 GiB | 5.62 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 581 ms | 574 ms–593 ms | 1.00× | 5.64 GiB | 5.63 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 539 ms | 528 ms–548 ms | 1.00× | 5.64 GiB | 5.64 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 12.5 s | 12.5 s–12.6 s | 1.00× | 3.52 GiB | 3.51 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 7.4 s | 7.33 s–7.59 s | 1.00× | 3.60 GiB | 3.59 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.67 s | 4.06 s–5.68 s | 1.00× | 3.76 GiB | 3.75 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.3 s | 2.05 s–3.08 s | 1.00× | 4.09 GiB | 4.08 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.63 s | 1.48 s–1.67 s | 1.00× | 4.41 GiB | 4.41 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.21 s | 1.2 s–1.25 s | 1.00× | 4.74 GiB | 4.73 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.08 s | 1.06 s–1.14 s | 1.00× | 5.06 GiB | 5.05 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 938 ms | 915 ms–965 ms | 1.00× | 5.38 GiB | 5.37 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 766 ms | 751 ms–788 ms | 1.00× | 5.62 GiB | 5.62 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 723 ms | 717 ms–756 ms | 1.00× | 5.63 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 706 ms | 679 ms–724 ms | 1.00× | 5.65 GiB | 5.64 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 683 ms | 665 ms–703 ms | 1.00× | 5.65 GiB | 5.65 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 657 ms | 636 ms–679 ms | 1.00× | 5.66 GiB | 5.66 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 666 ms | 634 ms–694 ms | 1.00× | 5.67 GiB | 5.66 GiB | 935 | 468 | 10 | 63.76% |  |
