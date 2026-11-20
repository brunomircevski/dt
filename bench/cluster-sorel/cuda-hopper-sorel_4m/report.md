## cuda-hopper-sorel_4m

**cuda-hopper-sorel_4m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 81bf6e6c8979, CPUs in order of use 16-31,0-15, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 639 MiB

### sorel_4m (4,000,000 train / 4,195,042 test rows, 2381 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 3.05 s | 3.02 s–3.1 s |  | 107.15 GiB (+ GPU 75.87 GiB) | 106.53 GiB | 1,619 | 810 | 10 | 91.92% | 92.57% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 3.37 s | 3.35 s–3.42 s |  | 110.95 GiB (+ GPU 75.84 GiB) | 110.32 GiB | 299 | 150 | 10 | 81.92% | 78.30% |

Test accuracy is measured on 4,195,042 rows: its 95% interval is ±0.04 percentage points at most.
