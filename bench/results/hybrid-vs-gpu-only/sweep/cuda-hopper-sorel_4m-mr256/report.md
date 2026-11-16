## cuda-hopper-sorel_4m-mr256

**cuda-hopper-sorel_4m-mr256**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree f53347a9d336, CPUs in order of use 16-31,0-15, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 643 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 22.4 s | 22.1 s–22.6 s |  | 107.23 GiB (+ GPU 76.25 GiB) | 106.61 GiB | 87,729 | 43,865 | 231 | 100.00% |  |

**C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows** (`c45_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 54.7 s | 54.7 s–54.8 s |  | 110.98 GiB (+ GPU 75.87 GiB) | 110.35 GiB | 69,637 | 34,819 | 572 | 99.81% |  |
