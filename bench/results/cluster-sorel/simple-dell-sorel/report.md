## simple-dell-sorel

**simple-dell-sorel**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 1007 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor powersave, turbo on, ./tree 8187eb9a778c-dirty, CPUs in order of use 0-47, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree simple 5 MiB

### sorel (12,699,013 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree simple | 1.24e+04 s | 1.24e+04 s–1.24e+04 s |  | 112.93 GiB | 112.93 GiB | 1,767 | 884 | 10 | 91.98% | 92.65% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree simple | 1.27e+04 s | 1.27e+04 s–1.27e+04 s |  | 123.04 GiB | 123.04 GiB | 379 | 190 | 10 | 81.73% | 78.11% |

Test accuracy is measured on 4,195,042 rows: its 95% interval is ±0.04 percentage points at most.
