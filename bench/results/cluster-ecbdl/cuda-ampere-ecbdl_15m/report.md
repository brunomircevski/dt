## cuda-ampere-ecbdl_15m

**cuda-ampere-ecbdl_15m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 8b1ecd151dd2, CPUs in order of use 0-47, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### ecbdl_15m (15,000,000 train / 100,000 test rows, 631 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 11.6 s | 11.6 s–11.6 s |  | 37.47 GiB (+ GPU 73.30 GiB) | 36.85 GiB | 1 | 1 | 0 | 97.85% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 65.1 s | 65.1 s–65.1 s |  | 107.83 GiB (+ GPU 73.35 GiB) | 107.20 GiB | 209,799 | 104,900 | 266 | 99.09% |  |
