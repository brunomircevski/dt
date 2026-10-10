## cuda-legion-higgs

**cuda-legion-higgs**: 13th Gen Intel(R) Core(TM) i7-13650HX, 20 logical CPUs (P-cores 0-11, E-cores 12-19), 31 GiB, Linux 7.2.9-1-cachyos #1 SMP PREEMPT_DYNAMIC Sat, 03 Oct 2026 18:34:55 +0000 x86_64, governor performance, turbo on, ./tree abc0a2212bba, CPUs in order of use 4,0-3,5-19, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB, ./tree CUDA 280 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.6 s | 4.6 s–4.6 s | 1.00× | 5.40 GiB | 5.40 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |
| ./tree CUDA | 2.06 s | 2.06 s–2.06 s | 0.45× | 3.83 GiB (+ GPU 4.67 GiB) | 3.56 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 6.65 s | 6.65 s–6.65 s | 1.00× | 7.63 GiB | 7.62 GiB | 1,160,123 | 580,062 | 71 | 87.89% | 70.10% |
| ./tree CUDA | 4.08 s | 4.08 s–4.08 s | 0.61× | 6.06 GiB (+ GPU 4.66 GiB) | 5.78 GiB | 1,160,123 | 580,062 | 71 | 87.89% | 70.10% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
