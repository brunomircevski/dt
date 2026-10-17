## cpu-epyc-ecbdl

**cpu-epyc-ecbdl**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree cd187178dd26-dirty, CPUs in order of use 0-47, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 7 MiB

### ecbdl (31,892,921 train / 100,000 test rows, 631 features, 2 classes)

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 84.3 s | 84.3 s–84.3 s | 1.00× | 237.01 GiB | 237.00 GiB | 1,205,683 | 602,842 | 111 | 100.00% |  |
