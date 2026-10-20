## cpu-epyc-ecbdl_15m

**cpu-epyc-ecbdl_15m**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 051949ba26af-dirty, CPUs in order of use 0-47, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB

### ecbdl_15m (15,000,000 train / 100,000 test rows, 631 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 16.8 s | 16.8 s–16.8 s | 1.00× | 111.45 GiB | 111.44 GiB | 1 | 1 | 0 | 97.85% |  |

**CART, grown to pure leaves, no pruning, no depth limit** (`cart_full_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 35.3 s | 35.3 s–35.3 s | 1.00× | 111.49 GiB | 111.48 GiB | 604,225 | 302,113 | 94 | 100.00% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 79.3 s | 79.3 s–79.3 s | 1.00× | 112.80 GiB | 112.79 GiB | 209,799 | 104,900 | 266 | 99.09% |  |
