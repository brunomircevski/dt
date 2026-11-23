## gpuonly-ampere-chess_1m

**gpuonly-ampere-chess_1m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-43-generic #43~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Sat Oct 31 06:52:47 UTC 2 x86_64, governor ondemand, turbo on, ./tree 7f98876f7e2a, CPUs in order of use 0-23,24-47, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree GPU 378 MiB

### chess_1m (1,000,000 train / 0 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 97 ms | 92 ms–104 ms |  | 388 MiB (+ GPU 485 MiB) | 10 MiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 97 ms | 91 ms–105 ms |  | 389 MiB (+ GPU 485 MiB) | 11 MiB | 17 | 9 | 4 | 100.00% |  |
