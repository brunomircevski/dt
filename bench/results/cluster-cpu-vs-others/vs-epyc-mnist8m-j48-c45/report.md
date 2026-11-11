## vs-epyc-mnist8m-j48-c45

**vs-epyc-mnist8m-j48-c45**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor performance, turbo on, ./tree f1fe270e76bb, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 98 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Weka J48 | 1.9e+04 s | 1.9e+04 s–1.9e+04 s |  | 81.64 GiB | 81.54 GiB | 1,453 | 727 | 10 | 70.96% | 74.65% |

Test accuracy is measured on 10,000 rows: its 95% interval is ±0.85 percentage points at most.
