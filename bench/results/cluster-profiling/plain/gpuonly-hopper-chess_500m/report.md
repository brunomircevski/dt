## gpuonly-hopper-chess_500m

**gpuonly-hopper-chess_500m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree ec77727, CPUs in order of use 16-31, 30 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 648 MiB

### chess_500m (500,000,000 train / 4,500 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 312 ms | 304 ms–326 ms |  | 5.30 GiB (+ GPU 13.30 GiB) | 4.67 GiB | 17 | 9 | 4 | 100.00% | 99.98% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 337 ms | 328 ms–347 ms |  | 5.30 GiB (+ GPU 13.30 GiB) | 4.67 GiB | 17 | 9 | 4 | 100.00% | 100.00% |

Test accuracy is measured on 4,500 rows: its 95% interval is ±0.04 percentage points at most.
