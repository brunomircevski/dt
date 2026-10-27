## simple-epyc-chess_10m

**simple-epyc-chess_10m**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree e418f651855f, CPUs in order of use 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 50 MiB, scikit-learn 120 MiB, ./tree 6 MiB, ./tree simple 5 MiB

### chess_10m (10,000,000 train / 4,500 test rows, 2 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 340 ms | 339 ms–340 ms | 1.00× | 490 MiB | 485 MiB | 17 | 9 | 4 | 100.00% |  |
| ./tree simple | 6.1 s | 6.09 s–6.11 s | 17.95× | 262 MiB | 257 MiB | 17 | 9 | 4 | 100.00% |  |
| scikit-learn | 6.23 s | 6.22 s–6.27 s | 18.33× | 586 MiB | 466 MiB | 17 | 9 | 4 | 100.00% | 99.98% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 81 ms | 72 ms–82 ms | 1.00× | 491 MiB | 485 MiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 495 ms | 493 ms–496 ms | 1.00× | 651 MiB | 646 MiB | 17 | 9 | 4 | 100.00% |  |
| ./tree simple | 7.56 s | 7.54 s–7.57 s | 15.27× | 350 MiB | 345 MiB | 17 | 9 | 4 | 100.00% |  |
| scikit-learn | 6.25 s | 6.24 s–6.26 s | 12.63× | 586 MiB | 466 MiB | 17 | 9 | 4 | 100.00% | 99.98% |
| Weka J48 | 36.5 s | 36.4 s–36.7 s | 73.70× | 6.04 GiB | 5.99 GiB | 17 | 9 | 4 | 100.00% | 100.00% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 108 ms | 107 ms–125 ms | 1.00× | 673 MiB | 667 MiB | 17 | 9 | 4 | 100.00% |  |

Test accuracy is measured on 4,500 rows: its 95% interval is ±0.04 percentage points at most.
