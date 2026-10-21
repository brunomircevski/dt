## simple-epyc-covertype

**simple-epyc-covertype**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree e418f651855f, CPUs in order of use 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 50 MiB, scikit-learn 120 MiB, ./tree 6 MiB, ./tree simple 5 MiB

### covertype (464,810 train / 116,202 test rows, 54 features, 7 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 858 ms | 857 ms–858 ms | 1.00× | 309 MiB | 303 MiB | 9,851 | 4,926 | 35 | 94.18% |  |
| ./tree simple | 7.6 s | 7.56 s–7.6 s | 8.86× | 115 MiB | 110 MiB | 9,851 | 4,926 | 35 | 94.18% |  |
| scikit-learn | 4.74 s | 4.74 s–4.74 s | 5.52× | 237 MiB | 117 MiB | 14,737 | 7,369 | 35 | 95.93% | 92.62% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 82 ms | 80 ms–110 ms | 1.00× | 398 MiB | 392 MiB | 9,851 | 4,926 | 35 | 94.18% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.13 s | 1.13 s–1.13 s | 1.00× | 318 MiB | 312 MiB | 26,377 | 13,189 | 59 | 98.66% |  |
| ./tree simple | 12.6 s | 12.5 s–12.6 s | 11.13× | 305 MiB | 300 MiB | 26,377 | 13,189 | 59 | 98.66% |  |
| scikit-learn | 4.11 s | 4.09 s–4.11 s | 3.64× | 237 MiB | 117 MiB | 31,839 | 15,920 | 36 | 98.91% | 94.26% |
| Weka J48 | 141 s | 140 s–143 s | 124.84× | 2.66 GiB | 2.61 GiB | 26,219 | 13,110 | 59 | 98.65% | 94.44% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 124 ms | 113 ms–140 ms | 1.00× | 411 MiB | 405 MiB | 26,377 | 13,189 | 59 | 98.66% |  |

Test accuracy is measured on 116,202 rows: its 95% interval is ±0.15 percentage points at most.
