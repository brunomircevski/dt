## cuda-hopper-airline

**cuda-hopper-airline**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 2ced36833c6c, CPUs in order of use 0-31, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 229 MiB

### airline (114,569,017 train / 500,000 test rows, 13 features, 2 classes)

**CART, grown to pure leaves (depth <= 30), no pruning** (`cart_full`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 4.66 s | 4.47 s–5.05 s |  | 18.81 GiB (+ GPU 23.94 GiB) | 18.59 GiB | 21,600,857 | 10,800,429 | 30 | 89.67% | 77.14% |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 7.15 s | 6.57 s–7.27 s |  | 24.90 GiB (+ GPU 23.94 GiB) | 24.68 GiB | 19,579,127 | 9,789,564 | 74 | 90.60% | 77.90% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.12 percentage points at most.
