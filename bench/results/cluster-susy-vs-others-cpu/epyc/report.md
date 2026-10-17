## epyc

**epyc**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 3cbdb6f9272f, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows; ./tree's one-thread cases rerun on 2026-10-17 (./tree 3bd3a4651449)

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 48 MiB, rpart 71 MiB, scikit-learn 120 MiB, ./tree 8 MiB, YaDT 5 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.98 s | 4.95 s–4.99 s | 1.00× | 1.61 GiB | 1.61 GiB | 723 | 362 | 19 | 79.68% | 79.65% |
| scikit-learn | 499 s | 499 s–500 s | 100.18× | 656 MiB | 536 MiB | 1,059 | 530 | 17 | 79.46% | 79.48% |
| rpart | 840 s | 744 s–847 s | 168.55× | 3.55 GiB | 3.48 GiB | 701 | 351 | 19 | 79.67% | 79.64% |

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 490 ms | 452 ms–564 ms | 1.00× | 1.63 GiB | 1.62 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 6.77 s | 6.72 s–6.82 s | 1.00× | 1.77 GiB | 1.76 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| Weka J48 | 855 s | 850 s–858 s | 126.27× | 9.46 GiB | 9.41 GiB | 15,479 | 7,740 | 38 | 80.68% | 79.71% |
| YaDT | 179 s | 156 s–180 s | 26.45× | 3.33 GiB | 3.32 GiB | 15,487 | 7,744 | 33 | 80.67% | 79.75% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 608 ms | 564 ms–616 ms | 1.00× | 1.87 GiB | 1.87 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| YaDT | 31.8 s | 29.6 s–32.2 s | 52.34× | 5.52 GiB | 5.51 GiB | 15,487 | 7,744 | 33 | 80.67% | 79.75% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
