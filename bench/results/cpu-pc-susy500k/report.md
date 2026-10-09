## cpu-pc-susy500k

**cpu-pc-susy500k**: Intel(R) Core(TM) i7-14700KF, 28 logical CPUs (P-cores 0-15, E-cores 16-27), 31 GiB, Linux 7.2.9-arch1-1 #1 SMP PREEMPT_DYNAMIC Sat, 03 Oct 2026 11:43:28 +0000 x86_64, governor performance, turbo on, ./tree 582251cbe428-dirty, CPUs in order of use 2,0,1,3-27, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 55 MiB, rpart 72 MiB, scikit-learn 126 MiB, ./tree 7 MiB, YaDT 5 MiB

### susy_500k (500,000 train / 100,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 851 ms | 851 ms–851 ms | 1.00× | 138 MiB | 131 MiB | 3,269 | 1,635 | 30 | 81.98% | 79.26% |
| scikit-learn | 16.9 s | 16.9 s–16.9 s | 19.84× | 188 MiB | 62 MiB | 3,473 | 1,737 | 25 | 81.36% | 78.94% |
| rpart | 9.48 s | 9.48 s–9.48 s | 11.13× | 425 MiB | 354 MiB | 2,863 | 1,432 | 29 | 81.75% | 79.32% |

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 114 ms | 114 ms–114 ms | 1.00× | 248 MiB | 240 MiB | 3,269 | 1,635 | 30 | 81.98% | 79.26% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 638 ms | 638 ms–638 ms | 1.00× | 203 MiB | 196 MiB | 3,579 | 1,790 | 27 | 80.96% | 79.26% |
| Weka J48 | 25.9 s | 25.9 s–25.9 s | 40.55× | 749 MiB | 694 MiB | 3,635 | 1,818 | 27 | 80.97% | 79.27% |
| YaDT | 10.4 s | 10.4 s–10.4 s | 16.26× | 758 MiB | 753 MiB | 3,533 | 1,767 | 27 | 80.94% | 79.29% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 137 ms | 137 ms–137 ms | 1.00× | 314 MiB | 306 MiB | 3,579 | 1,790 | 27 | 80.96% | 79.26% |
| YaDT | 2.87 s | 2.87 s–2.87 s | 20.93× | 1.10 GiB | 1.10 GiB | 3,533 | 1,767 | 27 | 80.94% | 79.29% |

Test accuracy is measured on 100,000 rows: its 95% interval is ±0.25 percentage points at most.
