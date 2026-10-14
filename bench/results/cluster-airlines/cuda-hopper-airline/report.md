## cuda-hopper-airline

**cuda-hopper-airline**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 5f4f955be4a2, CPUs in order of use 0-31, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB, ./tree CUDA 229 MiB

### airline (114,569,017 train / 500,000 test rows, 13 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 9.42 s | 9.38 s–9.83 s | 1.00× | 28.96 GiB | 28.96 GiB | 3,013 | 1,507 | 25 | 71.54% | 71.59% |
| ./tree CUDA | 2.25 s | 2.25 s–2.29 s | 0.24× | 17.22 GiB (+ GPU 23.94 GiB) | 17.00 GiB | 3,013 | 1,507 | 25 | 71.54% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 21.2 s | 21.1 s–21.8 s | 1.00× | 38.09 GiB | 38.08 GiB | 8,548,725 | 4,274,363 | 72 | 87.42% | 79.70% |
| ./tree CUDA | 14.2 s | 14.2 s–14.3 s | 0.67× | 26.30 GiB (+ GPU 23.94 GiB) | 26.07 GiB | 8,548,725 | 4,274,363 | 72 | 87.42% | 79.70% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
