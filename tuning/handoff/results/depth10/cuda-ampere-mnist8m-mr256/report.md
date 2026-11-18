## cuda-ampere-mnist8m-mr256

**cuda-ampere-mnist8m-mr256**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree b565df0e3c9a, CPUs in order of use 0-23,24-47, 5 run(s) per case, warm-up 0 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 630 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 3.09 s | 3.08 s–3.19 s |  | 71.68 GiB (+ GPU 49.19 GiB) | 71.07 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 3.11 s | 3.07 s–3.13 s |  | 71.66 GiB (+ GPU 48.84 GiB) | 71.04 GiB | 1,453 | 727 | 10 | 70.96% |  |
