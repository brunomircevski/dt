## cuda-hopper-chess_2b

**cuda-hopper-chess_2b**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree e4bc6a2fa573, CPUs in order of use 0-15, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 648 MiB

### chess_2b (2,000,000,000 train / 4,500 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 2.02 s | 2.02 s–2.02 s |  | 49.09 GiB (+ GPU 49.85 GiB) | 48.46 GiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 2.03 s | 2.02 s–2.03 s |  | 49.10 GiB (+ GPU 49.86 GiB) | 48.46 GiB | 17 | 9 | 4 | 100.00% |  |
