## gpuonly-ampere-mnist8m

**gpuonly-ampere-mnist8m**: AMD EPYC 7402 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-38-generic #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 14 16:37:11 UTC 2 x86_64, governor ondemand, turbo on, ./tree 713914b84454, CPUs in order of use 0-23,24-47, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree GPU 372 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 4.47 s | 4.4 s–4.51 s |  | 24.12 GiB (+ GPU 49.19 GiB) | 23.75 GiB | 2,047 | 1,024 | 10 | 80.77% | 84.51% |

Test accuracy is measured on 10,000 rows: its 95% interval is ±0.71 percentage points at most.
