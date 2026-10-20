## cuda-hopper-ecbdl_15m

**cuda-hopper-ecbdl_15m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 8b1ecd151dd2, CPUs in order of use 0-31, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 647 MiB

### ecbdl_15m (15,000,000 train / 100,000 test rows, 631 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 18.9 s | 18.9 s–18.9 s |  | 106.67 GiB (+ GPU 73.46 GiB) | 106.04 GiB | 604,225 | 302,113 | 94 | 100.00% |  |
