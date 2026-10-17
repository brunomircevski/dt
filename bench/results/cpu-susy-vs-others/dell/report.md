## dell

**dell**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 1007 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor powersave, turbo on, ./tree 3cbdb6f9272f, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 48 MiB, rpart 71 MiB, scikit-learn 120 MiB, ./tree 8 MiB, YaDT 5 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 7.97 s | 7.97 s–7.98 s | 1.00× | 1.08 GiB | 1.07 GiB | 723 | 362 | 19 | 79.68% | 79.65% |
| scikit-learn | 546 s | 545 s–546 s | 68.46× | 656 MiB | 536 MiB | 1,059 | 530 | 17 | 79.46% | 79.48% |
| rpart | 900 s | 900 s–901 s | 112.92× | 3.55 GiB | 3.48 GiB | 701 | 351 | 19 | 79.67% | 79.64% |

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 560 ms | 524 ms–632 ms | 1.00× | 1.63 GiB | 1.62 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 10.8 s | 10.7 s–10.8 s | 1.00× | 1.24 GiB | 1.23 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| Weka J48 | 1.34e+03 s | 1.34e+03 s–1.34e+03 s | 124.32× | 9.46 GiB | 9.41 GiB | 15,479 | 7,740 | 38 | 80.68% | 79.71% |
| YaDT | 215 s | 215 s–215 s | 19.99× | 3.33 GiB | 3.32 GiB | 15,487 | 7,744 | 33 | 80.67% | 79.75% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 716 ms | 707 ms–762 ms | 1.00× | 1.88 GiB | 1.87 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| YaDT | 30.9 s | 29.7 s–33.6 s | 43.18× | 5.59 GiB | 5.58 GiB | 15,487 | 7,744 | 33 | 80.67% | 79.75% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
