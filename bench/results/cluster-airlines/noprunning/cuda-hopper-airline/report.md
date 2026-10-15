## cuda-hopper-airline

**cuda-hopper-airline**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 35cf31543002-dirty, CPUs in order of use 0-31, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 357 MiB

### airline (114,569,017 train / 500,000 test rows, 13 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.37 s | 3.29 s–3.44 s |  | 20.85 GiB (+ GPU 23.86 GiB) | 20.50 GiB | 47,146,877 | 23,573,439 | 65 | 100.00% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.66 s | 3.56 s–4.06 s |  | 25.03 GiB (+ GPU 23.85 GiB) | 24.68 GiB | 19,579,127 | 9,789,564 | 74 | 90.60% |  |
