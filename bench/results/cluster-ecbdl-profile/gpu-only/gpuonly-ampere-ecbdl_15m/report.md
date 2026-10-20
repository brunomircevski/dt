## gpuonly-ampere-ecbdl_15m

**gpuonly-ampere-ecbdl_15m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 235c43175b46-dirty, CPUs in order of use 0-23, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 641 MiB

### ecbdl_15m (15,000,000 train / 100,000 test rows, 631 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 11.7 s | 11.7 s–11.7 s |  | 36.02 GiB (+ GPU 73.30 GiB) | 35.39 GiB | 1 | 1 | 0 | 97.85% |  |

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 34.9 s | 34.9 s–34.9 s |  | 36.55 GiB (+ GPU 74.26 GiB) | 35.92 GiB | 604,225 | 302,113 | 94 | 100.00% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 83 s | 83 s–83 s |  | 104.89 GiB (+ GPU 73.67 GiB) | 104.26 GiB | 209,799 | 104,900 | 266 | 99.09% |  |
