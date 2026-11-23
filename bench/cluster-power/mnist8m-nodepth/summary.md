### CART (gini), MNIST8M, no depth limit, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | 6 × 6 | 28.5 s ± 0.3% | 915 W | 510 W | 963 W | 26.1 kJ ± 0.4% | 14.5 kJ | 20.2 kJ (-22.4%) | CPU socket 0 7.67 kJ, CPU socket 1 7.6 kJ, Memory 2.89 kJ, Unused GPUs 5.85 kJ, Rest of the server 2.07 kJ |
| GPU | hopper | 6 × 12 | 12 s ± 0.8% | 679 W | 275 W | 868 W | 8.17 kJ ± 0.4% | 3.3 kJ | 6.31 kJ (-22.8%) | CPU socket 0 1.24 kJ, CPU socket 1 1.06 kJ, Memory 487 J, GPU in use 2.78 kJ, Unused GPUs 1.86 kJ, Rest of the server 733 J |
| Hybrid | hopper | 6 × 16 | 7.89 s ± 0.4% | 994 W | 589 W | 1209 W | 7.85 kJ ± 0.2% | 4.65 kJ | 6.63 kJ (-15.5%) | CPU socket 0 1.8 kJ, CPU socket 1 1.69 kJ, Memory 598 J, GPU in use 1.99 kJ, Unused GPUs 1.22 kJ, Rest of the server 547 J |

### C4.5 (gain ratio), MNIST8M, no depth limit, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | 6 × 5 | 34.3 s ± 0.4% | 912 W | 507 W | 970 W | 31.3 kJ ± 0.4% | 17.4 kJ | 24.2 kJ (-22.5%) | CPU socket 0 9.18 kJ, CPU socket 1 9.1 kJ, Memory 3.47 kJ, Unused GPUs 7.04 kJ, Rest of the server 2.49 kJ |
| GPU | hopper | 6 × 9 | 15.1 s ± 0.8% | 661 W | 257 W | 861 W | 10 kJ ± 1.0% | 3.88 kJ | 7.67 kJ (-23.3%) | CPU socket 0 1.52 kJ, CPU socket 1 1.29 kJ, Memory 599 J, GPU in use 3.34 kJ, Unused GPUs 2.33 kJ, Rest of the server 923 J |
| Hybrid | hopper | 6 × 14 | 9.56 s ± 0.3% | 937 W | 532 W | 1177 W | 8.95 kJ ± 0.2% | 5.08 kJ | 7.48 kJ (-16.4%) | CPU socket 0 1.98 kJ, CPU socket 1 1.77 kJ, Memory 653 J, GPU in use 2.42 kJ, Unused GPUs 1.47 kJ, Rest of the server 650 J |

### The meter model

Each node's meter, fitted over every reading of its job: reading(t) = mean over [t − D − W, t − D] of a + G_idle + b × CPU + GPU, where CPU = RAPL packages + DRAM (W), GPU = the NVML power of the GPU in use (W, counted once) and G_idle = the mean NVML power of the node's other GPUs over the job (measured, not fitted; none on epyc). Halves: the same fit on the first and on the second half of the job's readings.

| Node | job | meter | refresh | readings | D | W | model | first half | second half | rms | idle: model / meter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hopper | 792 | sys_W | 1.0 s | 8838 | 0.4 s | 1.25 s | 55 W + 154 W + 1.03·CPU + GPU | 54 W + 154 W + 1.03·CPU + GPU | 56 W + 154 W + 1.03·CPU + GPU | 16.7 W | 405 / 403 W |

### PSU energy: the model against the meter

E per tree from the model (used everywhere above) against: the model with each half's coefficients; the model fitted without any reading of this case's processes (held out: a load the model has not seen) and those held-out readings' mean residual ± its standard error; the meter's own readings over the same windows (held, shifted by D + W/2), which is exact only for windows much longer than the meter's refresh; and the mean residual (reading − model) of the readings whose averaging window lies inside the training windows, ± its standard error, relative to the mean power.

| Implementation | Node | Algorithm | E per tree (model) | halves | held out | held-out readings: mean residual | meter | readings inside | mean residual inside |
|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | CART | 26.1 kJ | -0.0% / +0.0% | -0.1% | 851: +4.5 ± 0.4 W | 26.1 kJ (+0.2%) | 781 | +3.0 ± 0.3 W (+0.3%) |
| Multicore | hopper | C4.5 | 31.3 kJ | -0.0% / +0.0% | -0.1% | 842: +1.2 ± 0.4 W | 31.3 kJ (+0.0%) | 782 | +1.1 ± 0.2 W (+0.1%) |
| GPU | hopper | CART | 8.17 kJ | -0.1% / +0.1% | -0.1% | 1086: +3.8 ± 0.4 W | 8.19 kJ (+0.2%) | 760 | +3.6 ± 0.3 W (+0.5%) |
| GPU | hopper | C4.5 | 10 kJ | -0.1% / +0.1% | +0.0% | 991: -0.9 ± 0.4 W | 9.96 kJ (-0.4%) | 744 | -0.0 ± 0.3 W (-0.0%) |
| Hybrid | hopper | CART | 7.85 kJ | -0.0% / +0.0% | +0.1% | 1062: -1.4 ± 0.8 W | 7.76 kJ (-1.1%) | 630 | -1.2 ± 1.0 W (-0.1%) |
| Hybrid | hopper | C4.5 | 8.95 kJ | -0.0% / +0.0% | +0.1% | 1084: -2.4 ± 0.7 W | 8.85 kJ (-1.1%) | 690 | -3.4 ± 0.9 W (-0.4%) |

### Time check

Time per tree (./tree's train time) against the first tree of each process and the time benchmark (hybrid-vs-gpu-only, mnist8m, cart_full_nodepth and c45_unpruned). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|---|
| Multicore | hopper | CART | 28.5 s | 28.6 s (-0.2%) | – |
| Multicore | hopper | C4.5 | 34.3 s | 34.4 s (-0.2%) | – |
| GPU | hopper | CART | 12 s | 12 s (+0.1%) | 12.3 s (-2.2%) |
| GPU | hopper | C4.5 | 15.1 s | 15.2 s (-0.5%) | 15.4 s (-1.7%) |
| Hybrid | hopper | CART | 7.89 s | 8.13 s (-2.9%) | 8.11 s (-2.7%) |
| Hybrid | hopper | C4.5 | 9.56 s | 9.44 s (+1.3%) | 9.56 s (-0.0%) |

### Warm-up check

GPU and Hybrid, CART, one process per warm-up (--repeat-warmup S: each tree's child keeps its CPU threads busy S s before its timer). Mean of trees 1–2 and of the rest.

| Node | Implementation | warm-up | trees | trees 1–2 | trees 3– | slowest |
|---|---|---|---|---|---|---|
| hopper | GPU | 0 s | 10 | 12.265 s | 12.227 s | 12.516 s |
| hopper | GPU | 2 s | 10 | 11.864 s | 12.010 s | 12.236 s |
| hopper | GPU | 15 s | 10 | 11.737 s | 11.755 s | 11.808 s |
| hopper | Hybrid | 0 s | 10 | 7.942 s | 7.834 s | 8.036 s |
| hopper | Hybrid | 2 s | 10 | 7.948 s | 7.852 s | 8.148 s |
| hopper | Hybrid | 15 s | 10 | 7.825 s | 7.846 s | 7.925 s |
