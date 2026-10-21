## simple-dell-covertype

**simple-dell-covertype**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 1007 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor powersave, turbo on, ./tree abd4bc744e17-dirty, CPUs in order of use 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 50 MiB, scikit-learn 120 MiB, ./tree 6 MiB, ./tree simple 5 MiB

### covertype (464,810 train / 116,202 test rows, 54 features, 7 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.1 s | 1.09 s–1.11 s | 1.00× | 309 MiB | 303 MiB | 9,851 | 4,926 | 35 | 94.18% |  |
| ./tree simple | 8.62 s | 8.62 s–8.64 s | 7.84× | 114 MiB | 109 MiB | 9,851 | 4,926 | 35 | 94.18% |  |
| scikit-learn | 4.31 s | 4.3 s–4.31 s | 3.92× | 238 MiB | 118 MiB | 48,085 | 24,043 | 44 | 100.00% | 93.89% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 103 ms | 101 ms–114 ms | 1.00× | 398 MiB | 392 MiB | 9,851 | 4,926 | 35 | 94.18% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.67 s | 1.67 s–1.68 s | 1.00× | 318 MiB | 312 MiB | 26,377 | 13,189 | 59 | 98.66% |  |
| ./tree simple | 14.2 s | 14.2 s–14.2 s | 8.48× | 305 MiB | 300 MiB | 26,377 | 13,189 | 59 | 98.66% |  |
| scikit-learn | 3.96 s | 3.96 s–3.96 s | 2.37× | 238 MiB | 118 MiB | 42,773 | 21,387 | 38 | 100.00% | 94.46% |
| Weka J48 | 187 s | 186 s–187 s | 111.56× | 2.66 GiB | 2.61 GiB | 26,219 | 13,110 | 59 | 98.65% | 94.44% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 180 ms | 172 ms–188 ms | 1.00× | 410 MiB | 404 MiB | 26,377 | 13,189 | 59 | 98.66% |  |

Test accuracy is measured on 116,202 rows: its 95% interval is ±0.14 percentage points at most.
