## cuda-hopper-higgs

**cuda-hopper-higgs**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 74a011fda9d3, CPUs in order of use 0-15, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 648 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 176 ms | 175 ms–177 ms |  | 3.96 GiB (+ GPU 3.33 GiB) | 3.32 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 191 ms | 186 ms–192 ms |  | 3.99 GiB (+ GPU 3.33 GiB) | 3.36 GiB | 935 | 468 | 10 | 63.76% | 63.73% |

**CART, grown to depth 10, pruned at a fixed alpha (cost-complexity)** (`cart_depth10_alpha`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 170 ms | 169 ms–172 ms |  | 1.77 GiB (+ GPU 3.33 GiB) | 1.14 GiB | 585 | 293 | 10 | 70.56% | 70.47% |

**C4.5, grown to depth 10, error-based pruning (CF 0.25, subtree raising)** (`c45_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 312 ms | 310 ms–316 ms |  | 4.09 GiB (+ GPU 4.47 GiB) | 3.46 GiB | 779 | 390 | 10 | 63.76% | 63.73% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
