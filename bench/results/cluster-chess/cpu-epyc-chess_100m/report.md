## cpu-epyc-chess_100m

**cpu-epyc-chess_100m**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 2a9d64ee830b, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB

### chess_100m (100,000,000 train / 4,500 test rows, 2 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.15 s | 1.14 s–1.16 s | 1.00× | 4.68 GiB | 4.68 GiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.35 s | 1.32 s–1.4 s | 1.00× | 4.68 GiB | 4.68 GiB | 17 | 9 | 4 | 100.00% |  |
