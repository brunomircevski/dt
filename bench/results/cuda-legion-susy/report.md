## cuda-legion-susy

**cuda-legion-susy**: 13th Gen Intel(R) Core(TM) i7-13650HX, 20 logical CPUs (P-cores 0-11, E-cores 12-19), 31 GiB, Linux 7.2.9-1-cachyos #1 SMP PREEMPT_DYNAMIC Sat, 03 Oct 2026 18:34:55 +0000 x86_64, governor performance, turbo on, ./tree a730d4a0183f, CPUs in order of use 4,0-3,5-19, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 7 MiB, ./tree CUDA 284 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 10.2 s | 10.2 s–10.2 s | 1.00× | 1.13 GiB | 1.12 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.3 s | 1.3 s–1.3 s | 1.00× | 1.74 GiB | 1.73 GiB | 723 | 362 | 19 | 79.68% | 79.65% |
| ./tree CUDA | 619 ms | 619 ms–619 ms | 0.48× | 1.26 GiB (+ GPU 1.40 GiB) | 1010 MiB | 723 | 362 | 19 | 79.68% | 79.65% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.45 s | 8.45 s–8.45 s | 1.00× | 1.72 GiB | 1.71 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.41 s | 1.41 s–1.41 s | 1.00× | 2.33 GiB | 2.32 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |
| ./tree CUDA | 819 ms | 819 ms–819 ms | 0.58× | 1.85 GiB (+ GPU 1.40 GiB) | 1.57 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
