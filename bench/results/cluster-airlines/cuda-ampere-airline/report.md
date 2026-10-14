## cuda-ampere-airline

**cuda-ampere-airline**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 5f4f955be4a2, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB, ./tree CUDA 222 MiB

### airline (114,569,017 train / 500,000 test rows, 13 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 16.4 s | 16.4 s–17.4 s | 1.00× | 28.97 GiB | 28.96 GiB | 3,013 | 1,507 | 25 | 71.54% | 71.59% |
| ./tree CUDA | 3.35 s | 3.32 s–3.36 s | 0.20× | 17.22 GiB (+ GPU 23.91 GiB) | 17.00 GiB | 3,013 | 1,507 | 25 | 71.54% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 33 s | 32.8 s–33.6 s | 1.00× | 38.06 GiB | 38.05 GiB | 8,548,725 | 4,274,363 | 72 | 87.42% | 79.70% |
| ./tree CUDA | 13.4 s | 13.3 s–13.5 s | 0.41× | 26.57 GiB (+ GPU 23.91 GiB) | 26.35 GiB | 8,548,725 | 4,274,363 | 72 | 87.42% | 79.70% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
