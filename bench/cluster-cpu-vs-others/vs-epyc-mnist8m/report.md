## vs-epyc-mnist8m

**vs-epyc-mnist8m**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor performance, turbo on, ./tree 81bf6e6c8979, CPUs in order of use 0-47, 4 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 98 MiB, scikit-learn 120 MiB, ./tree 6 MiB, YaDT 5 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 115 s | 115 s–116 s | 1.00× | 71.12 GiB | 71.12 GiB | 2,047 | 1,024 | 10 | 80.77% | 84.51% |
| scikit-learn | 1.01e+03 s | 1e+03 s–1.01e+03 s | 8.77× | 24.09 GiB | 23.97 GiB | 2,047 | 1,024 | 10 | 80.77% | 84.51% |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.25 s | 3.12 s–3.45 s | 1.00× | 74.02 GiB | 74.02 GiB | 2,047 | 1,024 | 10 | 80.77% | 84.51% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 114 s | 113 s–115 s | 1.00× | 71.13 GiB | 71.12 GiB | 1,453 | 727 | 10 | 70.96% | 74.65% |
| Weka J48 | 1.68e+04 s | 1.68e+04 s–1.69e+04 s | 147.30× | 81.55 GiB | 81.45 GiB | 1,453 | 727 | 10 | 70.96% | 74.65% |
| YaDT | 265 s | 264 s–266 s | 2.32× | 12.72 GiB | 12.72 GiB | 1,453 | 727 | 10 | 70.96% | 74.65% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.35 s | 3.27 s–3.86 s | 1.00× | 74.01 GiB | 74.00 GiB | 1,453 | 727 | 10 | 70.96% | 74.65% |
| YaDT | 9.77 s | 9.56 s–10 s | 2.91× | 19.65 GiB | 19.64 GiB | 1,453 | 727 | 10 | 70.96% | 74.65% |

Test accuracy is measured on 10,000 rows: its 95% interval is ±0.85 percentage points at most.
