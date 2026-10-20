## gpuonly-hopper-ecbdl_15m

**gpuonly-hopper-ecbdl_15m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree 7868bf4cb66d, CPUs in order of use 16-31, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 648 MiB

### ecbdl_15m (15,000,000 train / 100,000 test rows, 631 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 9.78 s | 9.78 s–9.78 s |  | 36.03 GiB (+ GPU 73.36 GiB) | 35.40 GiB | 1 | 1 | 0 | 97.85% |  |

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 28.7 s | 28.7 s–28.7 s |  | 36.56 GiB (+ GPU 74.30 GiB) | 35.93 GiB | 604,225 | 302,113 | 94 | 100.00% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 243 s | 243 s–243 s |  | 104.21 GiB (+ GPU 73.64 GiB) | 103.58 GiB | 209,799 | 104,900 | 266 | 99.09% |  |
