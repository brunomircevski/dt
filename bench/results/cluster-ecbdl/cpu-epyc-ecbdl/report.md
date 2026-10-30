## cpu-epyc-ecbdl

**cpu-epyc-ecbdl**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree ec77727, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### ecbdl (31,992,921 train / 2,897,917 test rows, 631 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 13.5 s | 13.5 s–13.6 s | 1.00× | 237.65 GiB | 237.65 GiB | 1,917 | 959 | 10 | 97.86% | 98.30% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 17.6 s | 17.5 s–17.7 s | 1.00× | 237.65 GiB | 237.64 GiB | 785 | 393 | 10 | 97.86% | 98.32% |

Test accuracy is measured on 2,897,917 rows: its 95% interval is ±0.01 percentage points at most.
