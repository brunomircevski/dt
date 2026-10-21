## simple-dell-chess_1m

**simple-dell-chess_1m**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 1007 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor powersave, turbo on, ./tree abd4bc744e17-dirty, CPUs in order of use 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 50 MiB, scikit-learn 120 MiB, ./tree 6 MiB, ./tree simple 5 MiB

### chess_1m (1,000,000 train / 1,000 test rows, 2 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 46 ms | 46 ms–46 ms | 1.00× | 55 MiB | 49 MiB | 17 | 9 | 4 | 100.00% |  |
| ./tree simple | 493 ms | 493 ms–495 ms | 10.73× | 32 MiB | 27 MiB | 17 | 9 | 4 | 100.00% |  |
| scikit-learn | 568 ms | 567 ms–570 ms | 12.35× | 166 MiB | 46 MiB | 17 | 9 | 4 | 100.00% | 100.00% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 54 ms | 48 ms–57 ms | 1.00× | 56 MiB | 50 MiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 72 ms | 72 ms–73 ms | 1.00× | 72 MiB | 66 MiB | 17 | 9 | 4 | 100.00% |  |
| ./tree simple | 640 ms | 640 ms–641 ms | 8.83× | 41 MiB | 36 MiB | 17 | 9 | 4 | 100.00% |  |
| scikit-learn | 577 ms | 576 ms–672 ms | 7.96× | 166 MiB | 46 MiB | 17 | 9 | 4 | 100.00% | 100.00% |
| Weka J48 | 2.71 s | 2.71 s–2.76 s | 37.36× | 904 MiB | 855 MiB | 17 | 9 | 4 | 100.00% | 100.00% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 73 ms | 63 ms–82 ms | 1.00× | 75 MiB | 69 MiB | 17 | 9 | 4 | 100.00% |  |
