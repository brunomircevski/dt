## epyc

**epyc**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 3cbdb6f9272f, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows; ./tree cases rerun on 2026-10-19, the other tools' rows are from 2026-10-16 (./tree cd187178dd26-dirty then)

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 48 MiB, rpart 71 MiB, scikit-learn 120 MiB, ./tree 7 MiB, YaDT 5 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.93 s | 4.93 s–4.97 s | 1.00× | 1.04 GiB | 1.04 GiB | 723 | 362 | 19 | 79.68% | 79.65% |
| scikit-learn | 499 s | 499 s–500 s | 101.18× | 656 MiB | 536 MiB | 1,059 | 530 | 17 | 79.46% | 79.48% |
| rpart | 840 s | 744 s–847 s | 170.23× | 3.55 GiB | 3.48 GiB | 701 | 351 | 19 | 79.67% | 79.64% |

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 486 ms | 476 ms–525 ms | 1.00× | 1.59 GiB | 1.59 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 6.39 s | 6.38 s–6.41 s | 1.00× | 1.20 GiB | 1.19 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| Weka J48 | 855 s | 850 s–858 s | 133.68× | 9.46 GiB | 9.41 GiB | 15,479 | 7,740 | 38 | 80.68% | 79.71% |
| YaDT | 179 s | 156 s–180 s | 28.00× | 3.33 GiB | 3.32 GiB | 15,487 | 7,744 | 33 | 80.67% | 79.75% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 575 ms | 538 ms–666 ms | 1.00× | 1.85 GiB | 1.85 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| YaDT | 31.8 s | 29.6 s–32.2 s | 55.37× | 5.52 GiB | 5.51 GiB | 15,487 | 7,744 | 33 | 80.67% | 79.75% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
