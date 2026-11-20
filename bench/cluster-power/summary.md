### CART (gini), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,291 s ± 0.0% | 256 W | 132 W | 285 W | 587 kJ ± 0.0% | 302 kJ | – | CPU socket 0 213 kJ, CPU socket 1 171 kJ, Rest of the server 203 kJ |
| Serial-Opt | epyc | 3 × 1 | 147 s ± 0.4% | 263 W | 138 W | 293 W | 38.5 kJ ± 0.9% | 20.3 kJ | – | CPU socket 0 14.2 kJ, CPU socket 1 11 kJ, Rest of the server 13.3 kJ |
| Multicore | epyc | 3 × 28 | 4.56 s ± 0.3% | 598 W | 474 W | 698 W | 2.73 kJ ± 0.3% | 2.16 kJ | – | CPU socket 0 973 J, CPU socket 1 914 J, Rest of the server 846 J |
| Serial-Simple | hopper | 1 × 1 | 2,551 s ± 0.0% | 542 W | 134 W | 549 W | 1.38 MJ ± 0.0% | 341 kJ | 863 kJ (-37.5%) | CPU socket 0 345 kJ, CPU socket 1 264 kJ, Memory 98.5 kJ, Unused GPUs 518 kJ, Rest of the server 156 kJ |
| Serial-Opt | hopper | 3 × 1 | 251 s ± 0.7% | 541 W | 133 W | 550 W | 136 kJ ± 0.7% | 33.5 kJ | 84.8 kJ (-37.6%) | CPU socket 0 33.4 kJ, CPU socket 1 25.8 kJ, Memory 10.2 kJ, Unused GPUs 51.1 kJ, Rest of the server 15.4 kJ |
| Multicore | hopper | 3 × 15 | 9.12 s ± 0.5% | 906 W | 498 W | 962 W | 8.26 kJ ± 0.3% | 4.54 kJ | 6.41 kJ (-22.5%) | CPU socket 0 2.45 kJ, CPU socket 1 2.36 kJ, Memory 1 kJ, Unused GPUs 1.86 kJ, Rest of the server 599 J |
| GPU | hopper | 3 × 49 | 2.54 s ± 1.1% | 740 W | 332 W | 977 W | 1.88 kJ ± 1.1% | 843 J | 1.49 kJ (-20.9%) | CPU socket 0 279 J, CPU socket 1 280 J, Memory 130 J, GPU in use 644 J, Unused GPUs 393 J, Rest of the server 152 J |
| Hybrid | hopper | 3 × 47 | 2.62 s ± 1.0% | 810 W | 402 W | 1168 W | 2.12 kJ ± 0.6% | 1.05 kJ | 1.72 kJ (-19.1%) | CPU socket 0 375 J, CPU socket 1 359 J, Memory 181 J, GPU in use 640 J, Unused GPUs 405 J, Rest of the server 161 J |

### C4.5 (gain ratio), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,596 s ± 0.0% | 257 W | 132 W | 289 W | 666 kJ ± 0.0% | 343 kJ | – | CPU socket 0 241 kJ, CPU socket 1 194 kJ, Rest of the server 230 kJ |
| Serial-Opt | epyc | 3 × 1 | 150 s ± 0.5% | 262 W | 138 W | 294 W | 39.3 kJ ± 1.2% | 20.7 kJ | – | CPU socket 0 14.5 kJ, CPU socket 1 11.3 kJ, Rest of the server 13.5 kJ |
| Multicore | epyc | 3 × 26 | 4.79 s ± 0.2% | 598 W | 473 W | 694 W | 2.87 kJ ± 0.6% | 2.27 kJ | – | CPU socket 0 1.01 kJ, CPU socket 1 972 J, Rest of the server 887 J |
| Serial-Simple | hopper | 1 × 1 | 2,793 s ± 0.0% | 542 W | 134 W | 582 W | 1.51 MJ ± 0.0% | 373 kJ | 945 kJ (-37.6%) | CPU socket 0 377 kJ, CPU socket 1 289 kJ, Memory 108 kJ, Unused GPUs 568 kJ, Rest of the server 171 kJ |
| Serial-Opt | hopper | 3 × 1 | 258 s ± 0.7% | 541 W | 133 W | 550 W | 139 kJ ± 0.7% | 34.3 kJ | 87 kJ (-37.6%) | CPU socket 0 34.3 kJ, CPU socket 1 26.4 kJ, Memory 10.5 kJ, Unused GPUs 52.4 kJ, Rest of the server 15.8 kJ |
| Multicore | hopper | 3 × 13 | 10.1 s ± 0.4% | 897 W | 489 W | 965 W | 9.08 kJ ± 0.3% | 4.95 kJ | 7.03 kJ (-22.7%) | CPU socket 0 2.69 kJ, CPU socket 1 2.57 kJ, Memory 1.11 kJ, Unused GPUs 2.06 kJ, Rest of the server 665 J |
| GPU | hopper | 3 × 49 | 2.55 s ± 0.3% | 738 W | 330 W | 870 W | 1.88 kJ ± 0.5% | 840 J | 1.49 kJ (-21.0%) | CPU socket 0 283 J, CPU socket 1 276 J, Memory 131 J, GPU in use 643 J, Unused GPUs 394 J, Rest of the server 154 J |
| Hybrid | hopper | 3 × 48 | 2.63 s ± 0.8% | 814 W | 406 W | 1110 W | 2.14 kJ ± 0.4% | 1.07 kJ | 1.73 kJ (-19.0%) | CPU socket 0 376 J, CPU socket 1 361 J, Memory 181 J, GPU in use 652 J, Unused GPUs 405 J, Rest of the server 162 J |

### The meter model

Each node's meter, fitted over every reading of its job: reading(t) = mean over [t − D − W, t − D] of a + G_idle + b × CPU + GPU, where CPU = RAPL packages + DRAM (W), GPU = the NVML power of the GPU in use (W, counted once) and G_idle = the mean NVML power of the node's other GPUs over the job (measured, not fitted; none on epyc). Halves: the same fit on the first and on the second half of the job's readings.

| Node | meter | refresh | readings | D | W | model | first half | second half | rms | idle: model / meter |
|---|---|---|---|---|---|---|---|---|---|---|
| epyc1 | sys_W | 6.3 s | 893 | 1.5 s | 2 s | 23 W + 1.39·CPU | 25 W + 1.37·CPU | 23 W + 1.40·CPU | 12.0 W | 124 / 119 W |
| hopper | sys_W | 1.0 s | 16113 | 0.5 s | 1 s | 58 W + 154 W + 1.01·CPU + GPU | 59 W + 154 W + 1.01·CPU + GPU | 56 W + 154 W + 1.02·CPU + GPU | 12.1 W | 408 / 406 W |

### PSU energy: the model against the meter

E per tree from the model (used everywhere above) against: the model with each half's coefficients; the model fitted without any reading of this case's processes (held out: a load the model has not seen) and those held-out readings' mean residual ± its standard error; the meter's own readings over the same windows (held, shifted by D + W/2), which is exact only for windows much longer than the meter's refresh; and the mean residual (reading − model) of the readings whose averaging window lies inside the training windows, ± its standard error, relative to the mean power.

| Implementation | Node | Algorithm | E per tree (model) | halves | held out | held-out readings: mean residual | meter | readings inside | mean residual inside |
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 587 kJ | -0.6% / +0.5% | -0.2% | 171: +3.0 ± 0.1 W | 592 kJ (+0.9%) | 167 | +2.4 ± 0.1 W (+0.9%) |
| Serial-Simple | epyc | C4.5 | 666 kJ | -0.6% / +0.5% | -0.3% | 185: +3.2 ± 0.1 W | 673 kJ (+1.0%) | 180 | +2.5 ± 0.1 W (+1.0%) |
| Serial-Opt | epyc | CART | 38.5 kJ | -0.7% / +0.5% | -0.2% | 78: +4.5 ± 0.2 W | 39.2 kJ (+1.8%) | 58 | +4.5 ± 0.2 W (+1.7%) |
| Serial-Opt | epyc | C4.5 | 39.3 kJ | -0.7% / +0.5% | -0.1% | 66: +2.9 ± 1.3 W | 40 kJ (+1.8%) | 51 | +4.6 ± 0.2 W (+1.8%) |
| Multicore | epyc | CART | 2.73 kJ | -1.4% / +0.5% | -0.5% | 84: +6.3 ± 2.6 W | 2.68 kJ (-1.8%) | 34 | -4.7 ± 1.8 W (-0.8%) |
| Multicore | epyc | C4.5 | 2.87 kJ | -1.4% / +0.5% | +0.1% | 81: +0.6 ± 1.4 W | 2.79 kJ (-2.7%) | 35 | -5.2 ± 2.0 W (-0.9%) |
| Serial-Simple | hopper | CART | 1.38 MJ | +0.0% / -0.1% | -0.0% | 1395: +0.0 ± 0.0 W | 1.38 MJ (-0.0%) | 1351 | -0.1 ± 0.0 W (-0.0%) |
| Serial-Simple | hopper | C4.5 | 1.51 MJ | +0.0% / -0.1% | -0.0% | 1513: +0.4 ± 0.0 W | 1.51 MJ (+0.0%) | 1464 | +0.2 ± 0.0 W (+0.0%) |
| Serial-Opt | hopper | CART | 136 kJ | +0.0% / -0.1% | -0.0% | 507: +1.3 ± 0.1 W | 136 kJ (-0.1%) | 361 | -0.3 ± 0.1 W (-0.0%) |
| Serial-Opt | hopper | C4.5 | 139 kJ | +0.0% / -0.1% | -0.0% | 583: +0.8 ± 0.1 W | 139 kJ (-0.2%) | 413 | -0.8 ± 0.1 W (-0.1%) |
| Multicore | hopper | CART | 8.26 kJ | -0.1% / +0.1% | -0.2% | 473: +7.7 ± 0.8 W | 8.25 kJ (-0.1%) | 340 | +5.3 ± 0.7 W (+0.6%) |
| Multicore | hopper | C4.5 | 9.08 kJ | -0.1% / +0.1% | -0.1% | 462: +5.9 ± 0.7 W | 9.06 kJ (-0.3%) | 339 | +3.8 ± 0.5 W (+0.4%) |
| GPU | hopper | CART | 1.88 kJ | +0.0% / -0.0% | +0.0% | 2199: -1.2 ± 0.2 W | 1.88 kJ (+0.3%) | 216 | -0.0 ± 1.1 W (-0.0%) |
| GPU | hopper | C4.5 | 1.88 kJ | +0.0% / -0.0% | +0.0% | 2208: -0.6 ± 0.2 W | 1.9 kJ (+1.0%) | 220 | +2.7 ± 1.0 W (+0.4%) |
| Hybrid | hopper | CART | 2.12 kJ | -0.0% / -0.0% | +0.0% | 2132: -1.2 ± 0.4 W | 2.11 kJ (-0.6%) | 233 | -2.5 ± 2.4 W (-0.3%) |
| Hybrid | hopper | C4.5 | 2.14 kJ | -0.0% / -0.0% | +0.0% | 2170: -1.4 ± 0.4 W | 2.13 kJ (-0.5%) | 239 | -3.9 ± 2.5 W (-0.5%) |

### Time check

Time per tree (./tree's train time) against the first tree of each process and the time benchmark (cluster-ecbdl, ecbdl_15m). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 2,291 s | 2,291 s (+0.0%) | 2,316 s (-1.1%) |
| Serial-Simple | epyc | C4.5 | 2,596 s | 2,596 s (+0.0%) | 2,628 s (-1.3%) |
| Serial-Opt | epyc | CART | 147 s | 147 s (+0.0%) | 148 s (-1.0%) |
| Serial-Opt | epyc | C4.5 | 150 s | 150 s (+0.0%) | 149 s (+1.0%) |
| Multicore | epyc | CART | 4.56 s | 4.5 s (+1.5%) | 4.47 s (+2.2%) |
| Multicore | epyc | C4.5 | 4.79 s | 4.82 s (-0.6%) | 4.73 s (+1.2%) |
| Serial-Simple | hopper | CART | 2,551 s | 2,551 s (+0.0%) | – |
| Serial-Simple | hopper | C4.5 | 2,793 s | 2,793 s (+0.0%) | – |
| Serial-Opt | hopper | CART | 251 s | 251 s (+0.0%) | – |
| Serial-Opt | hopper | C4.5 | 258 s | 258 s (+0.0%) | – |
| Multicore | hopper | CART | 9.12 s | 9.24 s (-1.3%) | – |
| Multicore | hopper | C4.5 | 10.1 s | 10.3 s (-1.6%) | – |
| GPU | hopper | CART | 2.54 s | 2.55 s (-0.6%) | 2.81 s (-9.8%) ! |
| GPU | hopper | C4.5 | 2.55 s | 2.55 s (-0.2%) | 2.84 s (-10.2%) ! |
| Hybrid | hopper | CART | 2.62 s | 2.65 s (-1.1%) | 2.94 s (-10.8%) ! |
| Hybrid | hopper | C4.5 | 2.63 s | 2.66 s (-1.3%) | 2.98 s (-11.9%) ! |

### Warm-up check

GPU and Hybrid, CART, one process per warm-up (--repeat-warmup S: each tree's child keeps its CPU threads busy S s before its timer). Mean of trees 1–2 and of the rest.

| Node | Implementation | warm-up | trees | trees 1–2 | trees 3– | slowest |
|---|---|---|---|---|---|---|
| hopper | GPU | 0 s | 12 | 2.598 s | 2.766 s | 2.847 s |
| hopper | GPU | 2 s | 22 | 2.703 s | 2.737 s | 2.885 s |
| hopper | GPU | 5 s | 12 | 2.544 s | 2.477 s | 2.551 s |
| hopper | GPU | 15 s | 22 | 2.482 s | 2.512 s | 2.626 s |
| hopper | Hybrid | 0 s | 12 | 2.694 s | 2.814 s | 2.840 s |
| hopper | Hybrid | 2 s | 22 | 2.754 s | 2.729 s | 3.014 s |
| hopper | Hybrid | 5 s | 12 | 2.478 s | 2.450 s | 2.488 s |
| hopper | Hybrid | 15 s | 22 | 2.580 s | 2.568 s | 2.750 s |
