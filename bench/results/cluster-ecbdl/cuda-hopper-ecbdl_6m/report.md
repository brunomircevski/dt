## cuda-hopper-ecbdl_6m

**cuda-hopper-ecbdl_6m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 49ebb6d007a6, CPUs in order of use 0-31, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 391 MiB

### ecbdl_6m (6,000,000 train / 100,000 test rows, 631 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 4.08 s | 4.08 s–4.08 s |  | 17.58 GiB (+ GPU 57.28 GiB) | 17.20 GiB | 1 | 1 | 0 | 97.85% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 13.9 s | 13.9 s–13.9 s |  | 43.02 GiB (+ GPU 71.46 GiB) | 42.64 GiB | 82,811 | 41,406 | 224 | 99.03% |  |
