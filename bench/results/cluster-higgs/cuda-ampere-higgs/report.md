## cuda-ampere-higgs

**cuda-ampere-higgs**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree f53347a9d336, CPUs in order of use 0-23,24-47, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 632 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 173 ms | 170 ms–185 ms |  | 3.94 GiB (+ GPU 3.30 GiB) | 3.32 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 194 ms | 192 ms–198 ms |  | 3.97 GiB (+ GPU 3.30 GiB) | 3.36 GiB | 935 | 468 | 10 | 63.76% | 63.73% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
