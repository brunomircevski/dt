## cuda-ampere-sorel_4m-mr16384

**cuda-ampere-sorel_4m-mr16384**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree b565df0e3c9a, CPUs in order of use 0-23,24-47, 5 run(s) per case, warm-up 0 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 630 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 4.4 s | 4.29 s–4.57 s |  | 107.12 GiB (+ GPU 75.80 GiB) | 106.50 GiB | 1,619 | 810 | 10 | 91.92% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 4.5 s | 4.43 s–4.6 s |  | 110.94 GiB (+ GPU 75.80 GiB) | 110.32 GiB | 299 | 150 | 10 | 81.92% |  |
