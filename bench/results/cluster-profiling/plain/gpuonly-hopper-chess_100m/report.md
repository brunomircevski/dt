## gpuonly-hopper-chess_100m

**gpuonly-hopper-chess_100m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree ec77727, CPUs in order of use 16-31, 30 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 648 MiB

### chess_100m (100,000,000 train / 4,500 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 87 ms | 82 ms–90 ms |  | 1.57 GiB (+ GPU 3.44 GiB) | 958 MiB | 17 | 9 | 4 | 100.00% | 99.98% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 91 ms | 83 ms–95 ms |  | 1.57 GiB (+ GPU 3.44 GiB) | 959 MiB | 17 | 9 | 4 | 100.00% | 100.00% |

Test accuracy is measured on 4,500 rows: its 95% interval is ±0.04 percentage points at most.
