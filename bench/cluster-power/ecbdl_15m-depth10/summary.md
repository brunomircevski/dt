### CART (gini), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 10 × 1 | 2,298 s ± 0.4% | 264 W | 126 W | 293 W | 606 kJ ± 1.2% | 289 kJ | – | CPU socket 0 213 kJ, CPU socket 1 172 kJ, Rest of the server 221 kJ |
| Serial-Opt | epyc | 10 × 1 | 147 s ± 0.3% | 265 W | 134 W | 296 W | 38.9 kJ ± 0.8% | 19.6 kJ | – | CPU socket 0 14.1 kJ, CPU socket 1 11 kJ, Rest of the server 13.8 kJ |
| Multicore | epyc | 3 × 28 | 4.56 s ± 0.3% | 598 W | 474 W | 698 W | 2.73 kJ ± 0.3% | 2.16 kJ | – | CPU socket 0 973 J, CPU socket 1 914 J, Rest of the server 846 J |
| Serial-Simple | hopper | 10 × 1 | 2,550 s ± 0.1% | 540 W | 135 W | 553 W | 1.38 MJ ± 0.1% | 345 kJ | 860 kJ (-37.5%) | CPU socket 0 344 kJ, CPU socket 1 264 kJ, Memory 98.3 kJ, Unused GPUs 516 kJ, Rest of the server 154 kJ |
| Serial-Opt | hopper | 10 × 1 | 251 s ± 0.7% | 539 W | 134 W | 556 W | 136 kJ ± 0.8% | 33.7 kJ | 84.6 kJ (-37.6%) | CPU socket 0 33.4 kJ, CPU socket 1 25.8 kJ, Memory 10.2 kJ, Unused GPUs 50.9 kJ, Rest of the server 15.2 kJ |
| Multicore | hopper | 3 × 15 | 9.13 s ± 0.3% | 906 W | 500 W | 956 W | 8.27 kJ ± 0.2% | 4.57 kJ | 6.43 kJ (-22.3%) | CPU socket 0 2.45 kJ, CPU socket 1 2.36 kJ, Memory 1 kJ, Unused GPUs 1.85 kJ, Rest of the server 616 J |
| GPU | hopper | 3 × 45 | 2.8 s ± 1.6% | 708 W | 306 W | 845 W | 1.98 kJ ± 0.9% | 856 J | 1.55 kJ (-21.7%) | CPU socket 0 281 J, CPU socket 1 285 J, Memory 142 J, GPU in use 676 J, Unused GPUs 430 J, Rest of the server 168 J |
| Hybrid | hopper | 3 × 41 | 2.92 s ± 0.3% | 714 W | 308 W | 813 W | 2.09 kJ ± 0.3% | 899 J | 1.64 kJ (-21.6%) | CPU socket 0 319 J, CPU socket 1 311 J, Memory 151 J, GPU in use 679 J, Unused GPUs 450 J, Rest of the server 177 J |

### C4.5 (gain ratio), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 10 × 1 | 2,599 s ± 0.2% | 264 W | 125 W | 291 W | 685 kJ ± 1.1% | 326 kJ | – | CPU socket 0 241 kJ, CPU socket 1 194 kJ, Rest of the server 250 kJ |
| Serial-Opt | epyc | 10 × 1 | 150 s ± 0.5% | 265 W | 134 W | 294 W | 39.8 kJ ± 1.0% | 20.1 kJ | – | CPU socket 0 14.4 kJ, CPU socket 1 11.2 kJ, Rest of the server 14.2 kJ |
| Multicore | epyc | 3 × 26 | 4.79 s ± 0.2% | 598 W | 473 W | 694 W | 2.87 kJ ± 0.6% | 2.27 kJ | – | CPU socket 0 1.01 kJ, CPU socket 1 972 J, Rest of the server 887 J |
| Serial-Simple | hopper | 10 × 1 | 2,785 s ± 0.1% | 540 W | 135 W | 554 W | 1.5 MJ ± 0.1% | 377 kJ | 939 kJ (-37.5%) | CPU socket 0 376 kJ, CPU socket 1 288 kJ, Memory 107 kJ, Unused GPUs 564 kJ, Rest of the server 168 kJ |
| Serial-Opt | hopper | 10 × 1 | 259 s ± 0.9% | 540 W | 134 W | 551 W | 140 kJ ± 0.9% | 34.8 kJ | 87.3 kJ (-37.5%) | CPU socket 0 34.5 kJ, CPU socket 1 26.6 kJ, Memory 10.5 kJ, Unused GPUs 52.5 kJ, Rest of the server 15.7 kJ |
| Multicore | hopper | 3 × 13 | 10.1 s ± 0.3% | 898 W | 491 W | 949 W | 9.1 kJ ± 0.2% | 4.98 kJ | 7.05 kJ (-22.6%) | CPU socket 0 2.69 kJ, CPU socket 1 2.57 kJ, Memory 1.11 kJ, Unused GPUs 2.05 kJ, Rest of the server 682 J |
| GPU | hopper | 3 × 44 | 2.84 s ± 1.2% | 714 W | 312 W | 856 W | 2.03 kJ ± 0.9% | 885 J | 1.59 kJ (-21.5%) | CPU socket 0 298 J, CPU socket 1 287 J, Memory 145 J, GPU in use 691 J, Unused GPUs 437 J, Rest of the server 170 J |
| Hybrid | hopper | 3 × 41 | 2.93 s ± 0.2% | 716 W | 309 W | 813 W | 2.1 kJ ± 0.3% | 906 J | 1.65 kJ (-21.5%) | CPU socket 0 318 J, CPU socket 1 310 J, Memory 149 J, GPU in use 692 J, Unused GPUs 451 J, Rest of the server 178 J |

### The meter model

Each node's meter, fitted over every reading of its job: reading(t) = mean over [t − D − W, t − D] of a + G_idle + b × CPU + GPU, where CPU = RAPL packages + DRAM (W), GPU = the NVML power of the GPU in use (W, counted once) and G_idle = the mean NVML power of the node's other GPUs over the job (measured, not fitted; none on epyc). Halves: the same fit on the first and on the second half of the job's readings.

| Node | job | meter | refresh | readings | D | W | model | first half | second half | rms | idle: model / meter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| epyc1 | 528 | sys_W | 6.3 s | 893 | 1.5 s | 2 s | 23 W + 1.39·CPU | 25 W + 1.37·CPU | 23 W + 1.40·CPU | 12.0 W | 124 / 119 W |
| epyc1 | 560 | sys_W | 6.3 s | 1248 | 1.4 s | 2.5 s | 36 W + 1.35·CPU | 53 W + 1.26·CPU | 10 W + 1.50·CPU | 7.9 W | 134 / 132 W |
| epyc1 | 657 | sys_W | 6.3 s | 2755 | 1.5 s | 2 s | 47 W + 1.30·CPU | 51 W + 1.28·CPU | 38 W + 1.36·CPU | 4.0 W | 141 / 137 W |
| hopper | 559 | sys_W | 1.0 s | 9782 | 0.5 s | 1 s | 57 W + 154 W + 1.02·CPU + GPU | 58 W + 154 W + 1.01·CPU + GPU | 57 W + 154 W + 1.02·CPU + GPU | 14.0 W | 406 / 404 W |
| hopper | 562 | sys_W | 1.0 s | 9540 | 0.5 s | 1 s | 58 W + 154 W + 1.01·CPU + GPU | 57 W + 154 W + 1.01·CPU + GPU | 61 W + 154 W + 1.00·CPU + GPU | 2.6 W | 405 / 404 W |
| hopper | 616 | sys_W | 1.0 s | 21160 | 0.5 s | 1 s | 57 W + 154 W + 1.01·CPU + GPU | 56 W + 154 W + 1.01·CPU + GPU | 57 W + 154 W + 1.01·CPU + GPU | 2.0 W | 404 / 403 W |
| hopper | 793 | sys_W | 1.0 s | 4897 | 0.4 s | 1.25 s | 53 W + 154 W + 1.03·CPU + GPU | 51 W + 154 W + 1.04·CPU + GPU | 56 W + 154 W + 1.02·CPU + GPU | 15.2 W | 402 / 401 W |
| hopper | 954 | sys_W | 1.0 s | 2926 | 0.3 s | 1.25 s | 58 W + 154 W + 1.01·CPU + GPU | 56 W + 154 W + 1.02·CPU + GPU | 61 W + 154 W + 1.00·CPU + GPU | 14.6 W | 407 / 405 W |

### PSU energy: the model against the meter

E per tree from the model (used everywhere above) against: the model with each half's coefficients; the model fitted without any reading of this case's processes (held out: a load the model has not seen) and those held-out readings' mean residual ± its standard error; the meter's own readings over the same windows (held, shifted by D + W/2), which is exact only for windows much longer than the meter's refresh; and the mean residual (reading − model) of the readings whose averaging window lies inside the training windows, ± its standard error, relative to the mean power.

| Implementation | Node | Algorithm | E per tree (model) | halves | held out | held-out readings: mean residual | meter | readings inside | mean residual inside |
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 606 kJ | +0.0% / +0.1% | -0.2% | 1697: +1.6 ± 0.1 W | 609 kJ (+0.5%) | 1633 | +1.2 ± 0.0 W (+0.5%) |
| Serial-Simple | epyc | C4.5 | 685 kJ | +0.0% / +0.1% | -0.3% | 1907: +1.9 ± 0.1 W | 688 kJ (+0.5%) | 1856 | +1.2 ± 0.0 W (+0.4%) |
| Serial-Opt | epyc | CART | 38.9 kJ | +0.3% / -0.0% | +0.0% | 250: +0.1 ± 0.5 W | 39 kJ (+0.3%) | 184 | +1.1 ± 0.3 W (+0.4%) |
| Serial-Opt | epyc | C4.5 | 39.8 kJ | +0.3% / -0.1% | +0.0% | 217: +0.1 ± 0.6 W | 39.9 kJ (+0.3%) | 162 | +0.8 ± 0.3 W (+0.3%) |
| Multicore | epyc | CART | 2.73 kJ | -1.4% / +0.5% | -0.5% | 84: +6.3 ± 2.6 W | 2.68 kJ (-1.8%) | 34 | -4.7 ± 1.8 W (-0.8%) |
| Multicore | epyc | C4.5 | 2.87 kJ | -1.4% / +0.5% | +0.1% | 81: +0.6 ± 1.4 W | 2.79 kJ (-2.7%) | 35 | -5.2 ± 2.0 W (-0.9%) |
| Serial-Simple | hopper | CART | 1.38 MJ | +0.0% / -0.0% | +0.0% | 13782: -0.5 ± 0.0 W | 1.38 MJ (-0.1%) | 13342 | -0.4 ± 0.0 W (-0.1%) |
| Serial-Simple | hopper | C4.5 | 1.5 MJ | +0.0% / -0.0% | -0.0% | 14678: +0.0 ± 0.0 W | 1.5 MJ (-0.0%) | 14240 | -0.1 ± 0.0 W (-0.0%) |
| Serial-Opt | hopper | CART | 136 kJ | -0.0% / -0.0% | +0.0% | 1748: -0.1 ± 0.1 W | 135 kJ (-0.2%) | 1303 | -1.2 ± 0.1 W (-0.2%) |
| Serial-Opt | hopper | C4.5 | 140 kJ | -0.0% / -0.0% | +0.0% | 1797: -0.3 ± 0.1 W | 139 kJ (-0.3%) | 1367 | -1.3 ± 0.1 W (-0.2%) |
| Multicore | hopper | CART | 8.27 kJ | -0.3% / +0.1% | -0.1% | 454: +1.9 ± 0.8 W | 8.21 kJ (-0.8%) | 339 | +1.0 ± 0.5 W (+0.1%) |
| Multicore | hopper | C4.5 | 9.1 kJ | -0.3% / +0.1% | -0.1% | 451: +3.6 ± 0.7 W | 9.05 kJ (-0.6%) | 338 | +1.5 ± 0.5 W (+0.2%) |
| GPU | hopper | CART | 1.98 kJ | +0.0% / -0.0% | +0.0% | 886: -0.8 ± 0.6 W | 1.96 kJ (-1.1%) | 212 | -2.2 ± 0.6 W (-0.3%) |
| GPU | hopper | C4.5 | 2.03 kJ | +0.0% / -0.0% | -0.0% | 867: +1.2 ± 0.6 W | 2.02 kJ (-0.5%) | 220 | -0.0 ± 0.7 W (-0.0%) |
| Hybrid | hopper | CART | 2.09 kJ | +0.0% / -0.0% | +0.1% | 808: -1.4 ± 0.7 W | 2.04 kJ (-2.2%) | 217 | +2.1 ± 1.0 W (+0.3%) |
| Hybrid | hopper | C4.5 | 2.1 kJ | +0.0% / -0.0% | +0.0% | 826: +0.2 ± 0.7 W | 2.06 kJ (-1.6%) | 230 | +2.6 ± 0.8 W (+0.4%) |

### Time check

Time per tree (./tree's train time) against the first tree of each process and the time benchmark (cluster-ecbdl, ecbdl_15m, cart_depth10 and c45_depth10_unpruned). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 2,298 s | 2,298 s (+0.0%) | 2,316 s (-0.8%) |
| Serial-Simple | epyc | C4.5 | 2,599 s | 2,599 s (+0.0%) | 2,628 s (-1.1%) |
| Serial-Opt | epyc | CART | 147 s | 147 s (+0.0%) | 148 s (-1.0%) |
| Serial-Opt | epyc | C4.5 | 150 s | 150 s (+0.0%) | 149 s (+1.2%) |
| Multicore | epyc | CART | 4.56 s | 4.5 s (+1.5%) | 4.47 s (+2.2%) |
| Multicore | epyc | C4.5 | 4.79 s | 4.82 s (-0.6%) | 4.73 s (+1.2%) |
| Serial-Simple | hopper | CART | 2,550 s | 2,550 s (+0.0%) | – |
| Serial-Simple | hopper | C4.5 | 2,785 s | 2,785 s (+0.0%) | – |
| Serial-Opt | hopper | CART | 251 s | 251 s (+0.0%) | – |
| Serial-Opt | hopper | C4.5 | 259 s | 259 s (+0.0%) | – |
| Multicore | hopper | CART | 9.13 s | 9.35 s (-2.4%) | – |
| Multicore | hopper | C4.5 | 10.1 s | 10.3 s (-2.0%) | – |
| GPU | hopper | CART | 2.8 s | 2.64 s (+6.2%) ! | 2.83 s (-1.1%) |
| GPU | hopper | C4.5 | 2.84 s | 2.9 s (-2.2%) | 2.86 s (-0.7%) |
| Hybrid | hopper | CART | 2.92 s | 2.94 s (-0.6%) | 2.94 s (-0.5%) |
| Hybrid | hopper | C4.5 | 2.93 s | 2.97 s (-1.1%) | 2.97 s (-1.2%) |

### Warm-up check

GPU and Hybrid, CART, one process per warm-up (--repeat-warmup S: each tree's child keeps its CPU threads busy S s before its timer). Mean of trees 1–2 and of the rest.

| Node | Implementation | warm-up | trees | trees 1–2 | trees 3– | slowest |
|---|---|---|---|---|---|---|
| hopper | GPU | 0 s | 20 | 2.936 s | 2.868 s | 2.989 s |
| hopper | GPU | 2 s | 20 | 2.869 s | 2.818 s | 2.889 s |
| hopper | GPU | 5 s | 20 | 2.539 s | 2.539 s | 2.590 s |
| hopper | GPU | 15 s | 20 | 2.542 s | 2.567 s | 2.669 s |
| hopper | Hybrid | 0 s | 20 | 3.077 s | 3.017 s | 3.126 s |
| hopper | Hybrid | 2 s | 20 | 2.948 s | 2.927 s | 3.018 s |
| hopper | Hybrid | 5 s | 20 | 2.650 s | 2.635 s | 2.727 s |
| hopper | Hybrid | 15 s | 20 | 2.716 s | 2.646 s | 2.849 s |
