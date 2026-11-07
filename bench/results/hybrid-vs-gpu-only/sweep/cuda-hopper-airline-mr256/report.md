## cuda-hopper-airline-mr256

**cuda-hopper-airline-mr256**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 95b6d774, CPUs in order of use 16-31,0-15, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 648 MiB

### airline (109,258,555 train / 5,810,462 test rows, 13 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.86 s | 3.85 s–4.32 s |  | 20.36 GiB (+ GPU 12.88 GiB) | 19.73 GiB | 44,663,207 | 22,331,604 | 66 | 100.00% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.94 s | 3.83 s–4.13 s |  | 19.82 GiB (+ GPU 12.82 GiB) | 19.18 GiB | 18,610,447 | 9,305,224 | 73 | 90.74% |  |
