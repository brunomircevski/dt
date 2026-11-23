## vs-hopper-higgs-depth30

**vs-hopper-higgs-depth30**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 7f98876f7e2a, CPUs in order of use 16-31,0-15, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree Hybrid 639 MiB, ./tree GPU 383 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 30, no pruning** (`cart_depth30`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree Hybrid | 818 ms | 739 ms–872 ms |  | 2.20 GiB (+ GPU 3.36 GiB) | 1.57 GiB | 2,158,499 | 1,079,250 | 30 | 95.73% | 68.77% |
| ./tree GPU | 999 ms | 943 ms–1.05 s |  | 2.02 GiB (+ GPU 3.74 GiB) | 1.65 GiB | 2,158,499 | 1,079,250 | 30 | 95.73% | 68.77% |
| cuML | 4.35 s | 3.64 s–4.55 s |  | 3.47 GiB (+ GPU 1.88 GiB) |  | 2,609,289 |  | 30 | 96.66% | 68.19% |
| XGBoost | 23.9 s | 23.7 s–23.9 s |  | 3.23 GiB (+ GPU 2.88 GiB) |  | 2,340,295 |  | 30 | 96.38% | 68.19% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
