## cpu-epyc-higgs

**cpu-epyc-higgs**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree ec77727, CPUs in order of use 0-47, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 10.4 s | 10.4 s–10.4 s | 1.00× | 3.57 GiB | 3.56 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 5.52 s | 5.5 s–5.53 s | 1.00× | 3.64 GiB | 3.64 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.84 s | 2.83 s–2.86 s | 1.00× | 3.80 GiB | 3.80 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.81 s | 1.78 s–1.91 s | 1.00× | 4.12 GiB | 4.11 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.29 s | 1.26 s–1.31 s | 1.00× | 4.43 GiB | 4.42 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.07 s | 1.04 s–1.14 s | 1.00× | 4.74 GiB | 4.74 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 946 ms | 932 ms–968 ms | 1.00× | 5.06 GiB | 5.05 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 831 ms | 815 ms–854 ms | 1.00× | 5.37 GiB | 5.37 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 820 ms | 797 ms–868 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 769 ms | 743 ms–824 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 726 ms | 687 ms–782 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 697 ms | 635 ms–773 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 678 ms | 594 ms–727 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 609 ms | 579 ms–633 ms | 1.00× | 5.61 GiB | 5.60 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 11.9 s | 11.9 s–12 s | 1.00× | 3.60 GiB | 3.59 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 6.32 s | 6.3 s–6.36 s | 1.00× | 3.68 GiB | 3.67 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.25 s | 3.22 s–3.26 s | 1.00× | 3.84 GiB | 3.83 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.9 s | 1.86 s–2.02 s | 1.00× | 4.15 GiB | 4.14 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.42 s | 1.39 s–1.42 s | 1.00× | 4.46 GiB | 4.46 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.13 s | 1.11 s–1.18 s | 1.00× | 4.78 GiB | 4.77 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.01 s | 995 ms–1.03 s | 1.00× | 5.09 GiB | 5.08 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 937 ms | 922 ms–956 ms | 1.00× | 5.40 GiB | 5.40 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 861 ms | 823 ms–898 ms | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 777 ms | 751 ms–847 ms | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 728 ms | 686 ms–851 ms | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 687 ms | 655 ms–822 ms | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 637 ms | 612 ms–809 ms | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 589 ms | 568 ms–628 ms | 1.00× | 5.64 GiB | 5.63 GiB | 935 | 468 | 10 | 63.76% | 63.73% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
