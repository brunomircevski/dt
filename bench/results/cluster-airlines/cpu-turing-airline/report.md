## cpu-turing-airline

**cpu-turing-airline**: AMD Ryzen Threadripper 1950X 16-Core Processor, 32 logical CPUs (P-cores None, E-cores None), 126 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor schedutil, turbo on, ./tree 971ad1433b82, CPUs in order of use 0-31, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB

### airline (114,569,017 train / 500,000 test rows, 13 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 38.1 s | 37.9 s–38.1 s | 1.00× | 28.96 GiB | 28.96 GiB | 3,013 | 1,507 | 25 | 71.54% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 64.1 s | 63.9 s–64.2 s | 1.00× | 38.08 GiB | 38.08 GiB | 8,548,725 | 4,274,363 | 72 | 87.42% | 79.70% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
