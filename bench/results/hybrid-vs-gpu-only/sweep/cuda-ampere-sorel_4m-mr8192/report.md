## cuda-ampere-sorel_4m-mr8192

**cuda-ampere-sorel_4m-mr8192**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree f53347a9d336, CPUs in order of use 0-23,24-47, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 632 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 29.2 s | 29.1 s–29.4 s |  | 107.13 GiB (+ GPU 75.83 GiB) | 106.51 GiB | 87,729 | 43,865 | 231 | 100.00% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 74.3 s | 74.2 s–74.5 s |  | 110.96 GiB (+ GPU 75.80 GiB) | 110.34 GiB | 69,637 | 34,819 | 572 | 99.81% |  |
