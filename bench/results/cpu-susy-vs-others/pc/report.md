## pc

**pc**: Intel(R) Core(TM) i7-14700KF, 28 logical CPUs (P-cores 0-15, E-cores 16-27), 31 GiB, Linux 7.2.9-arch1-1 #1 SMP PREEMPT_DYNAMIC Sat, 03 Oct 2026 11:43:28 +0000 x86_64, governor performance, turbo on, ./tree 49b7f472da08-dirty, CPUs in order of use 2,0,1,3-27, 1 run(s) per case, warm-up 50000 rows; ./tree cases rerun on 2026-10-16, the other tools' rows are from 2026-10-09 (./tree 49b7f472da08-dirty then)

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 55 MiB, rpart 72 MiB, scikit-learn 126 MiB, ./tree 7 MiB, YaDT 11 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.63 s | 4.63 s–4.63 s | 1.00× | 1.08 GiB | 1.07 GiB | 723 | 362 | 19 | 79.68% | 79.65% |
| scikit-learn | 412 s | 412 s–412 s | 89.00× | 662 MiB | 536 MiB | 1,059 | 530 | 17 | 79.46% | 79.48% |
| rpart | 362 s | 362 s–362 s | 78.11× | 3.55 GiB | 3.48 GiB | 701 | 351 | 19 | 79.67% | 79.64% |

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 801 ms | 799 ms–820 ms | 1.00× | 1.62 GiB | 1.62 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 5.99 s | 5.99 s–6.02 s | 1.00× | 1.24 GiB | 1.23 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| Weka J48 | 829 s | 829 s–829 s | 138.32× | 11.00 GiB | 10.94 GiB | 15,479 | 7,740 | 38 | 80.68% | 79.71% |
| YaDT | 126 s | 126 s–126 s | 21.09× | 3.32 GiB | 3.31 GiB | 15,487 | 7,744 | 33 | 80.67% | 79.75% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.19 s | 1.19 s–1.23 s | 1.00× | 1.89 GiB | 1.88 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| YaDT | 33.5 s | 33.5 s–33.5 s | 28.01× | 4.68 GiB | 4.67 GiB | 15,487 | 7,744 | 33 | 80.67% | 79.75% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
