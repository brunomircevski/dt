### CART (gini), MNIST8M, no depth limit, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | 6 × 6 | 28.5 s ± 0.4% | 913 W | 510 W | 958 W | 26 kJ ± 0.4% | 14.5 kJ | 20.2 kJ (-22.4%) | CPU socket 0 7.67 kJ, CPU socket 1 7.6 kJ, Memory 2.89 kJ, Unused GPUs 5.82 kJ, Rest of the server 2.05 kJ |
| GPU | hopper | 6 × 11 | 12.1 s ± 0.7% | 672 W | 268 W | 850 W | 8.15 kJ ± 0.5% | 3.26 kJ | 6.28 kJ (-22.9%) | CPU socket 0 1.22 kJ, CPU socket 1 1.05 kJ, Memory 490 J, GPU in use 2.77 kJ, Unused GPUs 1.87 kJ, Rest of the server 746 J |
| Hybrid | hopper | 6 × 17 | 7.82 s ± 0.3% | 1015 W | 611 W | 1222 W | 7.93 kJ ± 0.2% | 4.78 kJ | 6.73 kJ (-15.2%) | CPU socket 0 1.85 kJ, CPU socket 1 1.75 kJ, Memory 625 J, GPU in use 1.97 kJ, Unused GPUs 1.2 kJ, Rest of the server 539 J |

### C4.5 (gain ratio), MNIST8M, no depth limit, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | 6 × 5 | 34.4 s ± 0.2% | 910 W | 507 W | 965 W | 31.3 kJ ± 0.2% | 17.4 kJ | 24.2 kJ (-22.6%) | CPU socket 0 9.19 kJ, CPU socket 1 9.11 kJ, Memory 3.47 kJ, Unused GPUs 7.07 kJ, Rest of the server 2.44 kJ |
| GPU | hopper | 6 × 10 | 15.2 s ± 0.2% | 659 W | 255 W | 855 W | 9.99 kJ ± 0.5% | 3.87 kJ | 7.65 kJ (-23.4%) | CPU socket 0 1.48 kJ, CPU socket 1 1.27 kJ, Memory 601 J, GPU in use 3.36 kJ, Unused GPUs 2.34 kJ, Rest of the server 926 J |
| Hybrid | hopper | 6 × 14 | 9.52 s ± 0.5% | 954 W | 551 W | 1185 W | 9.08 kJ ± 0.2% | 5.24 kJ | 7.62 kJ (-16.1%) | CPU socket 0 2.04 kJ, CPU socket 1 1.83 kJ, Memory 686 J, GPU in use 2.42 kJ, Unused GPUs 1.46 kJ, Rest of the server 651 J |

### The meter model

Each node's meter, fitted over every reading of its job: reading(t) = mean over [t − D − W, t − D] of a + G_idle + b × CPU + GPU, where CPU = RAPL packages + DRAM (W), GPU = the NVML power of the GPU in use (W, counted once) and G_idle = the mean NVML power of the node's other GPUs over the job (measured, not fitted; none on epyc). Halves: the same fit on the first and on the second half of the job's readings.

| Node | job | meter | refresh | readings | D | W | model | first half | second half | rms | idle: model / meter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hopper | 561 | sys_W | 1.0 s | 8863 | 0.5 s | 1 s | 56 W + 154 W + 1.02·CPU + GPU | 56 W + 154 W + 1.02·CPU + GPU | 57 W + 154 W + 1.03·CPU + GPU | 17.3 W | 403 / 400 W |

### PSU energy: the model against the meter

E per tree from the model (used everywhere above) against: the model with each half's coefficients; the model fitted without any reading of this case's processes (held out: a load the model has not seen) and those held-out readings' mean residual ± its standard error; the meter's own readings over the same windows (held, shifted by D + W/2), which is exact only for windows much longer than the meter's refresh; and the mean residual (reading − model) of the readings whose averaging window lies inside the training windows, ± its standard error, relative to the mean power.

| Implementation | Node | Algorithm | E per tree (model) | halves | held out | held-out readings: mean residual | meter | readings inside | mean residual inside |
|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | CART | 26 kJ | -0.1% / +0.1% | -0.0% | 859: +0.4 ± 0.3 W | 26 kJ (-0.2%) | 803 | -0.2 ± 0.2 W (-0.0%) |
| Multicore | hopper | C4.5 | 31.3 kJ | -0.1% / +0.1% | -0.2% | 818: +8.0 ± 0.5 W | 31.5 kJ (+0.6%) | 764 | +5.4 ± 0.3 W (+0.6%) |
| GPU | hopper | CART | 8.15 kJ | -0.1% / +0.1% | -0.0% | 1006: -0.8 ± 0.4 W | 8.13 kJ (-0.2%) | 722 | +0.5 ± 0.3 W (+0.1%) |
| GPU | hopper | C4.5 | 9.99 kJ | -0.1% / +0.1% | -0.1% | 1103: +1.8 ± 0.4 W | 9.98 kJ (-0.1%) | 842 | +2.1 ± 0.3 W (+0.3%) |
| Hybrid | hopper | CART | 7.93 kJ | -0.1% / +0.1% | +0.0% | 1128: +1.3 ± 0.9 W | 7.83 kJ (-1.2%) | 687 | +0.2 ± 1.0 W (+0.0%) |
| Hybrid | hopper | C4.5 | 9.08 kJ | -0.1% / +0.1% | +0.1% | 1080: -4.0 ± 0.7 W | 8.97 kJ (-1.2%) | 718 | -5.7 ± 0.8 W (-0.6%) |

### Time check

Time per tree (./tree's train time) against the first tree of each process and the time benchmark (hybrid-vs-gpu-only, mnist8m, cart_full_nodepth and c45_unpruned). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|---|
| Multicore | hopper | CART | 28.5 s | 28.5 s (+0.2%) | – |
| Multicore | hopper | C4.5 | 34.4 s | 34.4 s (-0.2%) | – |
| GPU | hopper | CART | 12.1 s | 12.2 s (-0.3%) | 12.3 s (-1.3%) |
| GPU | hopper | C4.5 | 15.2 s | 15 s (+0.9%) | 15.3 s (-1.1%) |
| Hybrid | hopper | CART | 7.82 s | 8.05 s (-3.0%) | 8.04 s (-2.8%) |
| Hybrid | hopper | C4.5 | 9.52 s | 9.52 s (-0.0%) | 9.58 s (-0.6%) |

### Warm-up check

GPU and Hybrid, CART, one process per warm-up (--repeat-warmup S: each tree's child keeps its CPU threads busy S s before its timer). Mean of trees 1–2 and of the rest.

| Node | Implementation | warm-up | trees | trees 1–2 | trees 3– | slowest |
|---|---|---|---|---|---|---|
| hopper | GPU | 0 s | 10 | 12.143 s | 12.323 s | 12.514 s |
| hopper | GPU | 2 s | 10 | 11.960 s | 11.938 s | 12.306 s |
| hopper | GPU | 15 s | 10 | 11.729 s | 11.758 s | 11.875 s |
| hopper | Hybrid | 0 s | 10 | 7.881 s | 7.815 s | 7.999 s |
| hopper | Hybrid | 2 s | 10 | 7.850 s | 7.821 s | 7.975 s |
| hopper | Hybrid | 15 s | 10 | 7.691 s | 7.846 s | 8.005 s |
