## cuda-ampere-susy

**cuda-ampere-susy**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-43-generic #43~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Sat Oct 31 06:52:47 UTC 2 x86_64, governor ondemand, turbo on, ./tree 7f98876f7e2a, CPUs in order of use 0-23,24-47, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 633 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 279 ms | 265 ms–296 ms |  | 912 MiB (+ GPU 1.63 GiB) | 279 MiB | 1,959 | 980 | 10 | 79.16% | 79.01% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 287 ms | 266 ms–302 ms |  | 913 MiB (+ GPU 1.63 GiB) | 280 MiB | 743 | 372 | 10 | 78.74% | 78.65% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
