## simple-epyc-susy-sklearn-cart

**simple-epyc-susy-sklearn-cart**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 85ea70280e31, CPUs in order of use 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): scikit-learn 120 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scikit-learn | 853 s | 853 s–853 s |  | 754 MiB | 634 MiB | 1,059 | 530 | 17 | 79.46% | 79.48% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
