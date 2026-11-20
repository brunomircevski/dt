## vs-epyc-higgs

**vs-epyc-higgs**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor performance, turbo on, ./tree 81bf6e6c8979, CPUs in order of use 0-47, 5 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 98 MiB, rpart 71 MiB, scikit-learn 120 MiB, ./tree 6 MiB, YaDT 5 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 7.91 s | 7.89 s–7.92 s | 1.00× | 3.49 GiB | 3.48 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |
| scikit-learn | 216 s | 216 s–217 s | 27.30× | 1.61 GiB | 1.50 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |
| rpart | 153 s | 149 s–153 s | 19.28× | 8.24 GiB | 8.17 GiB | 1,181 | 591 | 10 | 70.63% | 70.52% |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 346 ms | 321 ms–393 ms | 1.00× | 5.64 GiB | 5.64 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 7.49 s | 7.47 s–7.5 s | 1.00× | 3.52 GiB | 3.51 GiB | 935 | 468 | 10 | 63.76% | 63.73% |
| Weka J48 | 770 s | 768 s–774 s | 102.88× | 12.49 GiB | 12.39 GiB | 923 | 462 | 10 | 63.77% | 63.75% |
| YaDT | 96.6 s | 95.3 s–97.2 s | 12.90× | 3.20 GiB | 3.20 GiB | 1,045 | 523 | 10 | 70.22% | 70.09% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 374 ms | 359 ms–447 ms | 1.00× | 5.67 GiB | 5.67 GiB | 935 | 468 | 10 | 63.76% | 63.73% |
| YaDT | 18.9 s | 17.4 s–20.1 s | 50.55× | 4.83 GiB | 4.83 GiB | 1,045 | 523 | 10 | 70.22% | 70.09% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
