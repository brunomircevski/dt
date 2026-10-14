## cuda-ampere-higgs

**cuda-ampere-higgs**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 5f4f955be4a2, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 7 MiB, ./tree CUDA 222 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.08 s | 2.05 s–2.09 s | 1.00× | 5.70 GiB | 5.70 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |
| ./tree CUDA | 601 ms | 584 ms–617 ms | 0.29× | 3.55 GiB (+ GPU 4.95 GiB) | 3.33 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.92 s | 3.83 s–3.93 s | 1.00× | 7.13 GiB | 7.12 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |
| ./tree CUDA | 1.43 s | 1.4 s–1.45 s | 0.36× | 5.00 GiB (+ GPU 4.95 GiB) | 4.78 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
