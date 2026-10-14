## cpu-pc-higgs

**cpu-pc-higgs**: Intel(R) Core(TM) i7-14700KF, 28 logical CPUs (P-cores 0-15, E-cores 16-27), 31 GiB, Linux 7.2.9-arch1-1 #1 SMP PREEMPT_DYNAMIC Sat, 03 Oct 2026 11:43:28 +0000 x86_64, governor performance, turbo on, ./tree 49b7f472da08-dirty, CPUs in order of use 2,0,1,3-27, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 7 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.47 s | 3.47 s–3.47 s | 1.00× | 5.92 GiB | 5.91 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 5.18 s | 5.18 s–5.18 s | 1.00× | 7.12 GiB | 7.11 GiB | 1,160,123 | 580,062 | 71 | 87.89% | 70.10% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
