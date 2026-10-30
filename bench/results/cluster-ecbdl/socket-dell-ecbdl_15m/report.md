## socket-dell-ecbdl_15m

**socket-dell-ecbdl_15m**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 1007 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor powersave, turbo on, ./tree ec77727, CPUs in order of use 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### ecbdl_15m (15,000,000 train / 2,897,917 test rows, 631 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 17.8 s | 17.7 s–17.9 s | 1.00× | 108.75 GiB | 108.75 GiB | 1,863 | 932 | 10 | 97.92% | 98.31% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 21.6 s | 21.5 s–21.7 s | 1.00× | 108.75 GiB | 108.74 GiB | 725 | 363 | 10 | 97.92% | 98.32% |

Test accuracy is measured on 2,897,917 rows: its 95% interval is ±0.01 percentage points at most.
