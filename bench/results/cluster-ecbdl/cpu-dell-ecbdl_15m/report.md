## cpu-dell-ecbdl_15m

**cpu-dell-ecbdl_15m**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 1007 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor powersave, turbo on, ./tree 74a011fda9d3, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### ecbdl_15m (15,000,000 train / 2,897,917 test rows, 631 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 10.5 s | 10.4 s–10.6 s | 1.00× | 111.44 GiB | 111.43 GiB | 1,863 | 932 | 10 | 97.92% | 98.31% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 12.3 s | 12.2 s–12.4 s | 1.00× | 111.43 GiB | 111.43 GiB | 725 | 363 | 10 | 97.92% | 98.32% |

**CART, grown to depth 10, pruned at a fixed alpha (cost-complexity)** (`cart_depth10_alpha`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 10.4 s | 10.3 s–10.6 s | 1.00× | 111.43 GiB | 111.43 GiB | 1 | 1 | 0 | 97.91% | 98.32% |

**C4.5, grown to depth 10, error-based pruning (CF 0.25, subtree raising)** (`c45_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 12.4 s | 12.3 s–12.5 s | 1.00× | 111.75 GiB | 111.74 GiB | 161 | 81 | 10 | 97.91% | 98.32% |

Test accuracy is measured on 2,897,917 rows: its 95% interval is ±0.01 percentage points at most.
