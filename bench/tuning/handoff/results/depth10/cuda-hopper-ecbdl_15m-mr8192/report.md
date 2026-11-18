## cuda-hopper-ecbdl_15m-mr8192

**cuda-hopper-ecbdl_15m-mr8192**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree b565df0e3c9a, CPUs in order of use 16-31,0-15, 5 run(s) per case, warm-up 0 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 641 MiB

### ecbdl_15m (15,000,000 train / 2,897,917 test rows, 631 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 2.73 s | 2.71 s–2.77 s |  | 106.50 GiB (+ GPU 73.49 GiB) | 105.87 GiB | 1,863 | 932 | 10 | 97.92% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 2.76 s | 2.74 s–2.77 s |  | 106.49 GiB (+ GPU 73.49 GiB) | 105.86 GiB | 725 | 363 | 10 | 97.92% |  |
