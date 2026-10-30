## vs-epyc-covertype-sklearn-cart

**vs-epyc-covertype-sklearn-cart**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree ec77727, CPUs in order of use 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): scikit-learn 120 MiB

### covertype (464,810 train / 116,202 test rows, 54 features, 7 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scikit-learn | 2.31 s | 2.31 s–2.32 s |  | 233 MiB | 113 MiB | 1,499 | 750 | 10 | 78.09% | 77.66% |

Test accuracy is measured on 116,202 rows: its 95% interval is ±0.24 percentage points at most.
