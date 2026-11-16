## serial-epyc-chess_100m

**serial-epyc-chess_100m**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor performance, turbo on, ./tree f53347a9d336, CPUs in order of use 0-47, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### chess_100m (100,000,000 train / 0 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.07 s | 2.07 s–2.07 s | 1.00× | 3.93 GiB | 3.93 GiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.41 s | 2.41 s–2.42 s | 1.00× | 3.93 GiB | 3.93 GiB | 17 | 9 | 4 | 100.00% |  |
