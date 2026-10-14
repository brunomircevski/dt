## cuda-hopper-higgs

**cuda-hopper-higgs**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree ab8ebdd36692, CPUs in order of use 0-31, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB, ./tree CUDA 229 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.19 s | 2.15 s–2.21 s | 1.00× | 5.91 GiB | 5.91 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |
| ./tree CUDA | 1.14 s | 1.14 s–1.17 s | 0.52× | 3.78 GiB (+ GPU 5.00 GiB) | 3.56 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.51 s | 3.5 s–3.91 s | 1.00× | 7.12 GiB | 7.11 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |
| ./tree CUDA | 2.33 s | 2.23 s–2.39 s | 0.66× | 4.98 GiB (+ GPU 4.99 GiB) | 4.76 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
