## cuda-hopper-susy-mr32768

**cuda-hopper-susy-mr32768**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 9885551da208, CPUs in order of use 0-15,16-31, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 642 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 61 ms | 57 ms–68 ms |  | 1.56 GiB (+ GPU 1.67 GiB) | 957 MiB | 1,959 | 980 | 10 | 79.16% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 70 ms | 67 ms–70 ms |  | 1.65 GiB (+ GPU 1.67 GiB) | 1.02 GiB | 743 | 372 | 10 | 78.74% |  |
