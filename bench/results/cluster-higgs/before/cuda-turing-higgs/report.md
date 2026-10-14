## cuda-turing-higgs

**cuda-turing-higgs**: AMD Ryzen Threadripper 1950X 16-Core Processor, 32 logical CPUs (P-cores None, E-cores None), 126 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor schedutil, turbo on, ./tree ae0eba0d89a6, CPUs in order of use 0-31, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB, ./tree CUDA 131 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 5.01 s | 4.99 s–5.02 s | 1.00× | 5.91 GiB | 5.91 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |
| ./tree CUDA | 2.76 s | 2.75 s–2.76 s | 0.55× | 3.68 GiB (+ GPU 4.75 GiB) | 3.56 GiB | 1,611 | 806 | 19 | 71.79% | 71.59% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 7.41 s | 7.4 s–7.44 s | 1.00× | 7.12 GiB | 7.11 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |
| ./tree CUDA | 4.42 s | 4.42 s–4.44 s | 0.60× | 4.88 GiB (+ GPU 4.74 GiB) | 4.76 GiB | 1,159,295 | 579,648 | 71 | 87.88% | 70.10% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
