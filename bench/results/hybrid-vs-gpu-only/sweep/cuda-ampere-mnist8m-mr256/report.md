## cuda-ampere-mnist8m-mr256

**cuda-ampere-mnist8m-mr256**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 95b6d774, CPUs in order of use 0-23,24-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA 641 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 23.2 s | 23.1 s–23.2 s |  | 71.97 GiB (+ GPU 49.41 GiB) | 71.34 GiB | 301,939 | 150,970 | 115 | 100.00% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA | 28.9 s | 28.8 s–29.2 s |  | 71.85 GiB (+ GPU 49.25 GiB) | 71.23 GiB | 218,325 | 109,163 | 147 | 99.61% |  |
