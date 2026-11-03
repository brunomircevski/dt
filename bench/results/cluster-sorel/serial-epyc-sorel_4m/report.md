## serial-epyc-sorel_4m

**serial-epyc-sorel_4m**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 4ece22f587b0, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 234 s | 234 s–234 s | 1.00× | 106.56 GiB | 106.56 GiB | 1,619 | 810 | 10 | 91.92% | 92.57% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 288 s | 288 s–288 s | 1.00× | 110.46 GiB | 110.45 GiB | 299 | 150 | 10 | 81.92% | 78.30% |

Test accuracy is measured on 4,195,042 rows: its 95% interval is ±0.04 percentage points at most.
