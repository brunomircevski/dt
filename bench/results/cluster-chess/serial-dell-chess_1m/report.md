## serial-dell-chess_1m

**serial-dell-chess_1m**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor powersave, turbo on, ./tree 82745fb51499, CPUs in order of use 0-47, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### chess_1m (1,000,000 train / 0 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 36 ms | 35 ms–36 ms | 1.00× | 47 MiB | 41 MiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 45 ms | 44 ms–46 ms | 1.00× | 48 MiB | 42 MiB | 17 | 9 | 4 | 100.00% |  |
