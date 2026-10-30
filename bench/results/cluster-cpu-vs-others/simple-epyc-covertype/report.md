## simple-epyc-covertype

**simple-epyc-covertype**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 67495f9-dirty, CPUs in order of use 0-47, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree simple 5 MiB

### covertype (464,810 train / 116,202 test rows, 54 features, 7 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree simple | 3.98 s | 3.98 s–3.99 s |  | 112 MiB | 107 MiB | 1,499 | 750 | 10 | 78.09% | 77.66% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree simple | 4.32 s | 4.31 s–4.32 s |  | 126 MiB | 122 MiB | 723 | 362 | 10 | 60.27% | 59.93% |

Test accuracy is measured on 116,202 rows: its 95% interval is ±0.28 percentage points at most.
