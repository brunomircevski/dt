## simple-epyc-covertype

**simple-epyc-covertype**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree abd4bc744e17-dirty, CPUs in order of use 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 50 MiB, scikit-learn 120 MiB, ./tree 6 MiB, ./tree simple 5 MiB

### covertype (464,810 train / 116,202 test rows, 54 features, 7 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 857 ms | 857 ms–857 ms | 1.00× | 309 MiB | 303 MiB | 9,851 | 4,926 | 35 | 94.18% |  |
| ./tree simple | 7.59 s | 7.58 s–7.62 s | 8.86× | 115 MiB | 110 MiB | 9,851 | 4,926 | 35 | 94.18% |  |
| scikit-learn | 4.23 s | 4.2 s–4.23 s | 4.94× | 238 MiB | 118 MiB | 48,085 | 24,043 | 44 | 100.00% | 93.89% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 82 ms | 82 ms–84 ms | 1.00× | 398 MiB | 392 MiB | 9,851 | 4,926 | 35 | 94.18% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.13 s | 1.13 s–1.13 s | 1.00× | 318 MiB | 312 MiB | 26,377 | 13,189 | 59 | 98.66% |  |
| ./tree simple | 12.5 s | 12.5 s–12.6 s | 11.12× | 305 MiB | 301 MiB | 26,377 | 13,189 | 59 | 98.66% |  |
| scikit-learn | 3.84 s | 3.81 s–3.84 s | 3.40× | 238 MiB | 118 MiB | 42,773 | 21,387 | 38 | 100.00% | 94.46% |
| Weka J48 | 142 s | 141 s–147 s | 125.61× | 2.66 GiB | 2.61 GiB | 26,219 | 13,110 | 59 | 98.65% | 94.44% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 122 ms | 119 ms–123 ms | 1.00× | 411 MiB | 405 MiB | 26,377 | 13,189 | 59 | 98.66% |  |

Test accuracy is measured on 116,202 rows: its 95% interval is ±0.14 percentage points at most.
