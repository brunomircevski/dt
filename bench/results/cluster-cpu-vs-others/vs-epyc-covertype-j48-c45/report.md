## vs-epyc-covertype-j48-c45

**vs-epyc-covertype-j48-c45**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree ec77727, CPUs in order of use 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): Weka J48 97 MiB

### covertype (464,810 train / 116,202 test rows, 54 features, 7 classes)

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Weka J48 | 36 s | 35.6 s–37.6 s |  | 1.14 GiB | 1.04 GiB | 723 | 362 | 10 | 60.27% | 59.93% |

Test accuracy is measured on 116,202 rows: its 95% interval is ±0.28 percentage points at most.
