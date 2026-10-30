## vs-epyc-susy-j48-c45

**vs-epyc-susy-j48-c45**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree ec77727, CPUs in order of use 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 97 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Weka J48 | 229 s | 229 s–230 s |  | 4.81 GiB | 4.71 GiB | 741 | 371 | 10 | 78.74% | 78.64% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
