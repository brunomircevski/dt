## simple-dell-ecbdl

**simple-dell-ecbdl**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor powersave, turbo on, ./tree 81bf6e6c8979, CPUs in order of use 0,4,8,12,16,20,24,28,32,36,40,44,2,6,10,14,18,22,26,30,34,38,42,46,1,5,9,13,17,21,25,29,33,37,41,45,3,7,11,15,19,23,27,31,35,39,43,47, 1 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree simple 5 MiB

### ecbdl (31,992,921 train / 2,897,917 test rows, 631 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree simple | 6.41e+03 s | 6.39e+03 s–6.44e+03 s |  | 75.72 GiB | 75.71 GiB | 1,917 | 959 | 10 | 97.86% | 98.30% |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree simple | 7.29e+03 s | 7.26e+03 s–7.32e+03 s |  | 75.88 GiB | 75.87 GiB | 785 | 393 | 10 | 97.86% | 98.32% |

Test accuracy is measured on 2,897,917 rows: its 95% interval is ±0.01 percentage points at most.
