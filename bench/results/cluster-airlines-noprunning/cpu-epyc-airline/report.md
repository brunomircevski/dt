## cpu-epyc-airline

**cpu-epyc-airline**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 2ced36833c6c, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB

### airline (114,569,017 train / 500,000 test rows, 13 features, 2 classes)

**CART, grown to pure leaves (depth <= 30), no pruning** (`cart_full`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.39 s | 8.38 s–8.44 s | 1.00× | 30.57 GiB | 30.56 GiB | 21,600,857 | 10,800,429 | 30 | 89.67% | 77.14% |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 11.1 s | 11 s–11.2 s | 1.00× | 36.66 GiB | 36.65 GiB | 19,579,127 | 9,789,564 | 74 | 90.60% | 77.90% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.12 percentage points at most.
