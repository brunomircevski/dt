## vs-hopper-higgs

**vs-hopper-higgs**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 7c39d6a3fc74, CPUs in order of use 16-31,0-15, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 638 MiB, ./tree GPU 383 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 327 ms | 284 ms–334 ms |  | 3.95 GiB (+ GPU 3.33 GiB) | 3.32 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |
| ./tree GPU | 216 ms | 198 ms–226 ms |  | 1.49 GiB (+ GPU 3.33 GiB) | 1.12 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |
| cuML | 550 ms | 497 ms–618 ms |  | 2.49 GiB (+ GPU 1.87 GiB) |  | 2,039 |  | 10 | 70.59% | 70.48% |
| XGBoost | 504 ms | 479 ms–543 ms |  | 2.28 GiB (+ GPU 2.73 GiB) |  | 2,031 |  | 10 | 70.63% | 70.53% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
