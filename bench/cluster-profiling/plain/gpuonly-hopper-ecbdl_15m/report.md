## gpuonly-hopper-ecbdl_15m

**gpuonly-hopper-ecbdl_15m**: Intel(R) Xeon(R) Gold 6444Y, 32 logical CPUs (P-cores None, E-cores None), 1008 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor schedutil, turbo on, ./tree 7f98876f7e2a, CPUs in order of use 16-31,0-15, 10 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree GPU 383 MiB

### ecbdl_15m (15,000,000 train / 2,897,917 test rows, 631 features, 2 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 2.83 s | 2.79 s–2.85 s |  | 35.73 GiB (+ GPU 72.99 GiB) | 35.35 GiB | 1,863 | 932 | 10 | 97.92% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree GPU | 2.86 s | 2.84 s–2.88 s |  | 35.71 GiB (+ GPU 72.99 GiB) | 35.33 GiB | 725 | 363 | 10 | 97.92% |  |
