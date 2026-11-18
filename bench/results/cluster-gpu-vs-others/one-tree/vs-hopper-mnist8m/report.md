## vs-hopper-mnist8m

**vs-hopper-mnist8m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 7c39d6a3fc74, CPUs in order of use 16-31,0-15, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 638 MiB, ./tree GPU 383 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 2.37 s | 2.35 s–2.39 s |  | 71.68 GiB (+ GPU 48.88 GiB) | 71.05 GiB | 2,047 | 1,024 | 10 | 80.77% | 84.51% |
| ./tree GPU | 2.25 s | 2.22 s–2.27 s |  | 24.13 GiB (+ GPU 49.22 GiB) | 23.75 GiB | 2,047 | 1,024 | 10 | 80.77% | 84.51% |
| cuML | 5.39 s | 5.21 s–5.49 s |  | 47.37 GiB (+ GPU 24.72 GiB) |  | 2,047 |  | 10 | 80.76% | 84.55% |
| XGBoost | 7.28 s | 7.22 s–7.58 s |  | 47.37 GiB (+ GPU 66.97 GiB) |  | 2,047 |  | 10 | 80.77% | 84.51% |

Test accuracy is measured on 10,000 rows: its 95% interval is ±0.71 percentage points at most.
