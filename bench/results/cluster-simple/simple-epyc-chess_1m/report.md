## simple-epyc-chess_1m

**simple-epyc-chess_1m**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree abd4bc744e17-dirty, CPUs in order of use 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 50 MiB, scikit-learn 119 MiB, ./tree 6 MiB, ./tree simple 5 MiB

### chess_1m (1,000,000 train / 1,000 test rows, 2 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 30 ms | 29 ms–30 ms | 1.00× | 55 MiB | 49 MiB | 17 | 9 | 4 | 100.00% |  |
| ./tree simple | 474 ms | 474 ms–475 ms | 16.07× | 32 MiB | 27 MiB | 17 | 9 | 4 | 100.00% |  |
| scikit-learn | 544 ms | 542 ms–556 ms | 18.43× | 166 MiB | 46 MiB | 17 | 9 | 4 | 100.00% | 100.00% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 22 ms | 21 ms–23 ms | 1.00× | 56 MiB | 50 MiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 45 ms | 45 ms–46 ms | 1.00× | 72 MiB | 66 MiB | 17 | 9 | 4 | 100.00% |  |
| ./tree simple | 623 ms | 622 ms–624 ms | 13.72× | 42 MiB | 37 MiB | 17 | 9 | 4 | 100.00% |  |
| scikit-learn | 548 ms | 545 ms–564 ms | 12.07× | 166 MiB | 46 MiB | 17 | 9 | 4 | 100.00% | 100.00% |
| Weka J48 | 2.69 s | 2.65 s–2.72 s | 59.36× | 904 MiB | 854 MiB | 17 | 9 | 4 | 100.00% | 100.00% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 30 ms | 29 ms–30 ms | 1.00× | 75 MiB | 69 MiB | 17 | 9 | 4 | 100.00% |  |
