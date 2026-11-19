## vs-epyc-higgs

**vs-epyc-higgs**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor performance, turbo on, ./tree 8fe0c73de5b7, CPUs in order of use 0-47, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 98 MiB, rpart 71 MiB, scikit-learn 120 MiB, YaDT 5 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| scikit-learn | 216 s | 214 s–217 s |  | 1.61 GiB | 1.50 GiB | 2,029 | 1,015 | 10 | 70.63% | 70.52% |
| rpart | 151 s | 149 s–153 s |  | 9.48 GiB | 9.41 GiB | 1,181 | 591 | 10 | 70.63% | 70.52% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Weka J48 | 770 s | 769 s–774 s |  | 12.49 GiB | 12.39 GiB | 923 | 462 | 10 | 63.77% | 63.75% |
| YaDT | 96.7 s | 96.1 s–97.6 s |  | 3.20 GiB | 3.20 GiB | 1,045 | 523 | 10 | 70.22% | 70.09% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| YaDT | 19.1 s | 17 s–21.2 s |  | 4.86 GiB | 4.85 GiB | 1,045 | 523 | 10 | 70.22% | 70.09% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
