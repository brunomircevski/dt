## simple-epyc-higgs-j48-c45

**simple-epyc-higgs-j48-c45**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 85ea70280e31, CPUs in order of use 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 51 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Weka J48 | 6.92e+04 s | 6.92e+04 s–6.92e+04 s |  | 44.93 GiB | 44.88 GiB | 1,152,747 | 576,374 | 71 | 87.83% | 70.15% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.13 percentage points at most.
