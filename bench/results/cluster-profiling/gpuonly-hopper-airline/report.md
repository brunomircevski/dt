## gpuonly-hopper-airline

**gpuonly-hopper-airline**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-34-generic #34~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 15:38:29 UTC 2 x86_64, governor schedutil, turbo on, ./tree e4bc6a2fa573, CPUs in order of use 0-15, 10 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree CUDA, GPU only 648 MiB

### airline (114,569,017 train / 500,000 test rows, 13 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 625 ms | 620 ms–635 ms |  | 6.41 GiB (+ GPU 13.27 GiB) | 5.78 GiB | 2,047 | 1,024 | 10 | 67.13% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree CUDA, GPU only | 644 ms | 638 ms–692 ms |  | 6.41 GiB (+ GPU 13.27 GiB) | 5.78 GiB | 905 | 453 | 10 | 66.88% |  |
