## cuda-turing-higgs

**cuda-turing-higgs**: AMD Ryzen Threadripper 1950X 16-Core Processor, 32 logical CPUs (P-cores None, E-cores None), 126 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor schedutil, turbo on, ./tree 5f4f955be4a2, CPUs in order of use 0-31, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB, ./tree CUDA 131 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.67 s | 3.63 s–3.67 s | 1.00× | 5.70 GiB | 5.69 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |
| ./tree CUDA | 1.91 s | 1.86 s–1.94 s | 0.52× | 3.46 GiB (+ GPU 4.73 GiB) | 3.33 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 6.78 s | 6.77 s–6.86 s | 1.00× | 7.12 GiB | 7.11 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |
| ./tree CUDA | 3.75 s | 3.69 s–3.85 s | 0.55× | 4.91 GiB (+ GPU 4.73 GiB) | 4.78 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
