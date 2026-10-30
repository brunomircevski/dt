## serial-epyc-chess_100m

**serial-epyc-chess_100m**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree ec77727, CPUs in order of use 0-47, 30 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### chess_100m (100,000,000 train / 4,500 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.21 s | 3.2 s–3.22 s | 1.00× | 4.68 GiB | 4.67 GiB | 17 | 9 | 4 | 100.00% | 99.98% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.59 s | 3.58 s–3.6 s | 1.00× | 4.68 GiB | 4.67 GiB | 17 | 9 | 4 | 100.00% | 100.00% |

Test accuracy is measured on 4,500 rows: its 95% interval is ±0.04 percentage points at most.
