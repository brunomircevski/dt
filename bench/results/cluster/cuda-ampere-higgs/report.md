## cuda-ampere-higgs

**cuda-ampere-higgs**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree ab8ebdd36692, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB, ./tree CUDA 222 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.42 s | 3.38 s–3.5 s | 1.00× | 5.91 GiB | 5.90 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |
| ./tree CUDA | 1.34 s | 1.31 s–1.39 s | 0.39× | 3.77 GiB (+ GPU 4.96 GiB) | 3.55 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 5.2 s | 5.19 s–5.3 s | 1.00× | 7.12 GiB | 7.12 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |
| ./tree CUDA | 2.48 s | 2.46 s–2.77 s | 0.48× | 4.97 GiB (+ GPU 4.96 GiB) | 4.75 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
