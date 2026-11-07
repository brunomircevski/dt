## cuda-hopper-mnist8m-mr2048

**cuda-hopper-mnist8m-mr2048**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 95b6d774, CPUs in order of use 16-31,0-15, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 647 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 16.1 s | 16.1 s–16.1 s |  | 71.73 GiB (+ GPU 48.94 GiB) | 71.10 GiB | 301,939 | 150,970 | 115 | 100.00% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 20.1 s | 19.9 s–20.2 s |  | 71.73 GiB (+ GPU 48.94 GiB) | 71.10 GiB | 218,325 | 109,163 | 147 | 99.61% |  |
