## cuda-hopper-sorel_4m-mr16384

**cuda-hopper-sorel_4m-mr16384**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 0b6ff5094490, CPUs in order of use 16-31,0-15, 5 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 638 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to depth 25, no pruning** (`cart_depth25`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 5.99 s | 5.87 s–6.02 s |  | 36.08 GiB (+ GPU 75.87 GiB) | 35.45 GiB | 34,045 | 17,023 | 25 | 98.64% |  |

**C4.5, grown to depth 25, unpruned (only C4.5's collapse of useless splits)** (`c45_depth25_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 5.88 s | 5.88 s–5.9 s |  | 36.08 GiB (+ GPU 75.84 GiB) | 35.45 GiB | 2,701 | 1,351 | 25 | 87.11% |  |
