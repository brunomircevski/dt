## simple-epyc-higgs

**simple-epyc-higgs**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor performance, turbo on, ./tree f1fe270e76bb, CPUs in order of use 0-47, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree simple 5 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree simple | 238 s | 235 s–238 s |  | 1.32 GiB | 1.32 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree simple | 259 s | 256 s–260 s |  | 1.43 GiB | 1.43 GiB | 935 | 468 | 10 | 63.76% | 63.73% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
