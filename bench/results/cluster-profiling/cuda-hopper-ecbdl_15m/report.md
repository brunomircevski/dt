## cuda-hopper-ecbdl_15m

**cuda-hopper-ecbdl_15m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree e4bc6a2fa573, CPUs in order of use 0-15, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 648 MiB

### ecbdl_15m (15,000,000 train / 100,000 test rows, 631 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.77 s | 3.77 s–3.78 s |  | 106.54 GiB (+ GPU 73.33 GiB) | 105.90 GiB | 1,841 | 921 | 10 | 97.86% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 3.88 s | 3.87 s–3.88 s |  | 106.52 GiB (+ GPU 73.33 GiB) | 105.89 GiB | 719 | 360 | 10 | 97.86% |  |
