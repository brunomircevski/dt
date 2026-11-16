## cuda-hopper-chess_500m

**cuda-hopper-chess_500m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree f53347a9d336, CPUs in order of use 16-31,0-15, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 643 MiB

### chess_500m (500,000,000 train / 0 test rows, 2 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 286 ms | 274 ms–288 ms |  | 12.75 GiB (+ GPU 13.30 GiB) | 12.12 GiB | 17 | 9 | 4 | 100.00% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 291 ms | 282 ms–304 ms |  | 12.75 GiB (+ GPU 13.30 GiB) | 12.12 GiB | 17 | 9 | 4 | 100.00% |  |
