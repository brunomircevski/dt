## vs-epyc-susy-sklearn-cart

**vs-epyc-susy-sklearn-cart**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree ec77727, CPUs in order of use 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): scikit-learn 120 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scikit-learn | 86.2 s | 86.2 s–87.1 s |  | 608 MiB | 488 MiB | 1,959 | 980 | 10 | 79.16% | 79.01% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
