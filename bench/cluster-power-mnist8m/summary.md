### CART (gini), MNIST8M, no depth limit, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | 6 × 6 | 28.5 s ± 0.2% | 915 W | 500 W | 964 W | 26.1 kJ ± 0.2% | 14.3 kJ | 20.2 kJ (-22.4%) | CPU socket 0 7.68 kJ, CPU socket 1 7.6 kJ, Memory 2.89 kJ, Unused GPUs 5.85 kJ, Rest of the server 2.06 kJ |
| GPU | hopper | 6 × 12 | 11.8 s ± 0.3% | 694 W | 279 W | 855 W | 8.2 kJ ± 0.4% | 3.29 kJ | 6.37 kJ (-22.3%) | CPU socket 0 1.33 kJ, CPU socket 1 1.13 kJ, Memory 481 J, GPU in use 2.72 kJ, Unused GPUs 1.82 kJ, Rest of the server 713 J |
| Hybrid | hopper | 6 × 17 | 7.8 s ± 0.3% | 1015 W | 600 W | 1198 W | 7.92 kJ ± 0.3% | 4.68 kJ | 6.72 kJ (-15.2%) | CPU socket 0 1.85 kJ, CPU socket 1 1.76 kJ, Memory 624 J, GPU in use 1.94 kJ, Unused GPUs 1.21 kJ, Rest of the server 540 J |

### C4.5 (gain ratio), MNIST8M, no depth limit, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | 6 × 5 | 34.4 s ± 0.3% | 911 W | 496 W | 961 W | 31.3 kJ ± 0.2% | 17.1 kJ | 24.3 kJ (-22.6%) | CPU socket 0 9.19 kJ, CPU socket 1 9.12 kJ, Memory 3.48 kJ, Unused GPUs 7.07 kJ, Rest of the server 2.48 kJ |
| GPU | hopper | 6 × 10 | 14.6 s ± 0.2% | 689 W | 274 W | 894 W | 10.1 kJ ± 0.5% | 4 kJ | 7.81 kJ (-22.5%) | CPU socket 0 1.65 kJ, CPU socket 1 1.39 kJ, Memory 584 J, GPU in use 3.31 kJ, Unused GPUs 2.26 kJ, Rest of the server 878 J |
| Hybrid | hopper | 6 × 14 | 9.28 s ± 0.4% | 981 W | 566 W | 1206 W | 9.11 kJ ± 0.1% | 5.25 kJ | 7.67 kJ (-15.8%) | CPU socket 0 2.04 kJ, CPU socket 1 1.95 kJ, Memory 674 J, GPU in use 2.38 kJ, Unused GPUs 1.44 kJ, Rest of the server 632 J |

### The meter model

Each node's meter, fitted over every reading of its job: reading(t) = mean over [t − D − W, t − D] of a + G_idle + b × CPU + GPU, where CPU = RAPL packages + DRAM (W), GPU = the NVML power of the GPU in use (W, counted once) and G_idle = the mean NVML power of the node's other GPUs over the job (measured, not fitted; none on epyc). Halves: the same fit on the first and on the second half of the job's readings.

| Node | meter | refresh | readings | D | W | model | first half | second half | rms | idle: model / meter |
|---|---|---|---|---|---|---|---|---|---|---|
| hopper | sys_W | 1.0 s | 10673 | 0.4 s | 1.25 s | 53 W + 155 W + 1.03·CPU + GPU | 54 W + 155 W + 1.03·CPU + GPU | 51 W + 155 W + 1.03·CPU + GPU | 18.9 W | 415 / 419 W |

### PSU energy: the model against the meter

E per tree from the model (used everywhere above) against: the model with each half's coefficients; the model fitted without any reading of this case's processes (held out: a load the model has not seen) and those held-out readings' mean residual ± its standard error; the meter's own readings over the same windows (held, shifted by D + W/2), which is exact only for windows much longer than the meter's refresh; and the mean residual (reading − model) of the readings whose averaging window lies inside the training windows, ± its standard error, relative to the mean power.

| Implementation | Node | Algorithm | E per tree (model) | halves | held out | held-out readings: mean residual | meter | readings inside | mean residual inside |
|---|---|---|---|---|---|---|---|---|---|
| Multicore | hopper | CART | 26.1 kJ | -0.0% / +0.1% | -0.1% | 820: +2.1 ± 1.2 W | 25.9 kJ (-0.9%) | 759 | +1.9 ± 1.3 W (+0.2%) |
| Multicore | hopper | C4.5 | 31.3 kJ | -0.0% / +0.1% | -0.2% | 825: +6.0 ± 0.5 W | 31.5 kJ (+0.4%) | 764 | +5.2 ± 0.2 W (+0.6%) |
| GPU | hopper | CART | 8.2 kJ | +0.1% / -0.1% | +0.1% | 1738: -2.5 ± 0.3 W | 8.09 kJ (-1.3%) | 739 | +0.2 ± 0.3 W (+0.0%) |
| GPU | hopper | C4.5 | 10.1 kJ | +0.1% / -0.1% | -0.0% | 1628: -0.4 ± 0.3 W | 10.1 kJ (+0.2%) | 788 | +1.6 ± 0.3 W (+0.2%) |
| Hybrid | hopper | CART | 7.92 kJ | -0.0% / +0.0% | +0.1% | 1960: -2.6 ± 0.5 W | 7.82 kJ (-1.2%) | 657 | +0.3 ± 1.0 W (+0.0%) |
| Hybrid | hopper | C4.5 | 9.11 kJ | -0.0% / +0.0% | +0.0% | 1820: -1.2 ± 0.4 W | 9.08 kJ (-0.2%) | 669 | +2.7 ± 0.9 W (+0.3%) |

### Time check

Time per tree (./tree's train time) against the first tree of each process and the time benchmark (none for MNIST8M, no depth limit). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|---|
| Multicore | hopper | CART | 28.5 s | 28.6 s (-0.2%) | – |
| Multicore | hopper | C4.5 | 34.4 s | 34.4 s (-0.2%) | – |
| GPU | hopper | CART | 11.8 s | 11.8 s (+0.2%) | – |
| GPU | hopper | C4.5 | 14.6 s | 14.7 s (-0.7%) | – |
| Hybrid | hopper | CART | 7.8 s | 7.81 s (-0.1%) | – |
| Hybrid | hopper | C4.5 | 9.28 s | 9.29 s (-0.1%) | – |
