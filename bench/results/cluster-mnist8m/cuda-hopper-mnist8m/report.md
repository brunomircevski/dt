## cuda-hopper-mnist8m

**cuda-hopper-mnist8m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree f53347a9d336, CPUs in order of use 16-31,0-15, 5 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 643 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 1.96 s | 1.92 s–2.17 s |  | 71.70 GiB (+ GPU 49.22 GiB) | 71.07 GiB | 2,047 | 1,024 | 10 | 80.77% | 84.51% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 2.18 s | 1.92 s–2.21 s |  | 71.66 GiB (+ GPU 48.85 GiB) | 71.03 GiB | 1,453 | 727 | 10 | 70.96% | 74.65% |

Test accuracy is measured on 10,000 rows: its 95% interval is ±0.85 percentage points at most.
