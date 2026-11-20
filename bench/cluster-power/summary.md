### CART (gini), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,291 s ± 0.0% | 256 W | 132 W | 285 W | 587 kJ ± 0.0% | 302 kJ | – | CPU socket 0 213 kJ, CPU socket 1 171 kJ, Rest of the server 203 kJ |
| Serial-Opt | epyc | 3 × 1 | 147 s ± 0.4% | 263 W | 138 W | 293 W | 38.5 kJ ± 0.9% | 20.3 kJ | – | CPU socket 0 14.2 kJ, CPU socket 1 11 kJ, Rest of the server 13.3 kJ |
| Multicore | epyc | 3 × 28 | 4.56 s ± 0.3% | 598 W | 474 W | 698 W | 2.73 kJ ± 0.3% | 2.16 kJ | – | CPU socket 0 973 J, CPU socket 1 914 J, Rest of the server 846 J |
| Serial-Simple | hopper | 1 × 1 | 2,549 s ± 0.0% | 541 W | 135 W | 549 W | 1.38 MJ ± 0.0% | 344 kJ | 863 kJ (-37.4%) | CPU socket 0 344 kJ, CPU socket 1 264 kJ, Memory 98.2 kJ, Unused GPUs 516 kJ, Rest of the server 157 kJ |
| Serial-Opt | hopper | 3 × 1 | 252 s ± 0.4% | 541 W | 134 W | 549 W | 136 kJ ± 0.4% | 33.9 kJ | 85.2 kJ (-37.5%) | CPU socket 0 33.5 kJ, CPU socket 1 25.8 kJ, Memory 10.3 kJ, Unused GPUs 51.1 kJ, Rest of the server 15.6 kJ |
| Multicore | hopper | 3 × 15 | 9.13 s ± 0.3% | 906 W | 500 W | 956 W | 8.27 kJ ± 0.2% | 4.57 kJ | 6.43 kJ (-22.3%) | CPU socket 0 2.45 kJ, CPU socket 1 2.36 kJ, Memory 1 kJ, Unused GPUs 1.85 kJ, Rest of the server 616 J |
| GPU | hopper | 3 × 44 | 2.82 s ± 1.6% | 704 W | 298 W | 833 W | 1.99 kJ ± 1.4% | 840 J | 1.55 kJ (-21.9%) | CPU socket 0 280 J, CPU socket 1 286 J, Memory 143 J, GPU in use 672 J, Unused GPUs 435 J, Rest of the server 169 J |
| Hybrid | hopper | 3 × 41 | 2.94 s ± 1.1% | 770 W | 363 W | 1072 W | 2.26 kJ ± 0.7% | 1.07 kJ | 1.81 kJ (-20.0%) | CPU socket 0 380 J, CPU socket 1 365 J, Memory 196 J, GPU in use 689 J, Unused GPUs 453 J, Rest of the server 181 J |

### C4.5 (gain ratio), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,596 s ± 0.0% | 257 W | 132 W | 289 W | 666 kJ ± 0.0% | 343 kJ | – | CPU socket 0 241 kJ, CPU socket 1 194 kJ, Rest of the server 230 kJ |
| Serial-Opt | epyc | 3 × 1 | 150 s ± 0.5% | 262 W | 138 W | 294 W | 39.3 kJ ± 1.2% | 20.7 kJ | – | CPU socket 0 14.5 kJ, CPU socket 1 11.3 kJ, Rest of the server 13.5 kJ |
| Multicore | epyc | 3 × 26 | 4.79 s ± 0.2% | 598 W | 473 W | 694 W | 2.87 kJ ± 0.6% | 2.27 kJ | – | CPU socket 0 1.01 kJ, CPU socket 1 972 J, Rest of the server 887 J |
| Serial-Simple | hopper | 1 × 1 | 2,783 s ± 0.0% | 541 W | 135 W | 549 W | 1.51 MJ ± 0.0% | 375 kJ | 942 kJ (-37.4%) | CPU socket 0 376 kJ, CPU socket 1 288 kJ, Memory 107 kJ, Unused GPUs 563 kJ, Rest of the server 172 kJ |
| Serial-Opt | hopper | 3 × 1 | 260 s ± 0.3% | 541 W | 135 W | 549 W | 141 kJ ± 0.3% | 35 kJ | 87.8 kJ (-37.5%) | CPU socket 0 34.5 kJ, CPU socket 1 26.7 kJ, Memory 10.6 kJ, Unused GPUs 52.7 kJ, Rest of the server 16 kJ |
| Multicore | hopper | 3 × 13 | 10.1 s ± 0.3% | 898 W | 491 W | 949 W | 9.1 kJ ± 0.2% | 4.98 kJ | 7.05 kJ (-22.6%) | CPU socket 0 2.69 kJ, CPU socket 1 2.57 kJ, Memory 1.11 kJ, Unused GPUs 2.05 kJ, Rest of the server 682 J |
| GPU | hopper | 3 × 44 | 2.84 s ± 1.3% | 707 W | 301 W | 916 W | 2.01 kJ ± 0.9% | 855 J | 1.57 kJ (-21.8%) | CPU socket 0 292 J, CPU socket 1 288 J, Memory 145 J, GPU in use 673 J, Unused GPUs 438 J, Rest of the server 172 J |
| Hybrid | hopper | 3 × 41 | 2.92 s ± 0.3% | 772 W | 366 W | 1144 W | 2.26 kJ ± 0.3% | 1.07 kJ | 1.81 kJ (-19.9%) | CPU socket 0 379 J, CPU socket 1 362 J, Memory 195 J, GPU in use 688 J, Unused GPUs 450 J, Rest of the server 181 J |

### The meter model

Each node's meter, fitted over every reading of its job: reading(t) = mean over [t − D − W, t − D] of a + G_idle + b × CPU + GPU, where CPU = RAPL packages + DRAM (W), GPU = the NVML power of the GPU in use (W, counted once) and G_idle = the mean NVML power of the node's other GPUs over the job (measured, not fitted; none on epyc). Halves: the same fit on the first and on the second half of the job's readings.

| Node | meter | refresh | readings | D | W | model | first half | second half | rms | idle: model / meter |
|---|---|---|---|---|---|---|---|---|---|---|
| epyc1 | sys_W | 6.3 s | 893 | 1.5 s | 2 s | 23 W + 1.39·CPU | 25 W + 1.37·CPU | 23 W + 1.40·CPU | 12.0 W | 124 / 119 W |
| hopper | sys_W | 1.0 s | 9782 | 0.5 s | 1 s | 57 W + 154 W + 1.02·CPU + GPU | 58 W + 154 W + 1.01·CPU + GPU | 57 W + 154 W + 1.02·CPU + GPU | 14.0 W | 406 / 404 W |

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
| Serial-Simple | hopper | CART | 1.38 MJ | -0.1% / +0.1% | +0.0% | 1361: -1.6 ± 0.1 W | 1.38 MJ (-0.3%) | 1317 | -1.6 ± 0.0 W (-0.3%) |
| Serial-Simple | hopper | C4.5 | 1.51 MJ | -0.1% / +0.1% | +0.0% | 1432: -1.3 ± 0.0 W | 1.5 MJ (-0.2%) | 1386 | -1.3 ± 0.0 W (-0.2%) |
| Serial-Opt | hopper | CART | 136 kJ | -0.1% / +0.1% | +0.0% | 545: -1.5 ± 0.1 W | 136 kJ (-0.5%) | 405 | -2.9 ± 0.1 W (-0.5%) |
| Serial-Opt | hopper | C4.5 | 141 kJ | -0.1% / +0.1% | +0.0% | 548: -1.1 ± 0.1 W | 140 kJ (-0.4%) | 412 | -2.3 ± 0.1 W (-0.4%) |
| Multicore | hopper | CART | 8.27 kJ | -0.3% / +0.1% | -0.1% | 454: +1.9 ± 0.8 W | 8.21 kJ (-0.8%) | 339 | +1.0 ± 0.5 W (+0.1%) |
| Multicore | hopper | C4.5 | 9.1 kJ | -0.3% / +0.1% | -0.1% | 451: +3.6 ± 0.7 W | 9.05 kJ (-0.6%) | 338 | +1.5 ± 0.5 W (+0.2%) |
| GPU | hopper | CART | 1.99 kJ | -0.0% / +0.0% | -0.0% | 825: +2.4 ± 0.6 W | 1.97 kJ (-0.6%) | 229 | +2.8 ± 0.7 W (+0.4%) |
| GPU | hopper | C4.5 | 2.01 kJ | -0.0% / +0.0% | -0.0% | 830: +2.2 ± 0.6 W | 2 kJ (-0.2%) | 235 | +4.5 ± 0.9 W (+0.6%) |
| Hybrid | hopper | CART | 2.26 kJ | -0.1% / +0.1% | +0.0% | 821: -0.2 ± 0.9 W | 2.23 kJ (-1.3%) | 238 | -3.2 ± 2.4 W (-0.4%) |
| Hybrid | hopper | C4.5 | 2.26 kJ | -0.1% / +0.1% | +0.0% | 805: +0.1 ± 0.9 W | 2.23 kJ (-1.1%) | 234 | -1.4 ± 2.3 W (-0.2%) |

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
| Serial-Simple | hopper | CART | 2,549 s | 2,549 s (+0.0%) | – |
| Serial-Simple | hopper | C4.5 | 2,783 s | 2,783 s (+0.0%) | – |
| Serial-Opt | hopper | CART | 252 s | 252 s (+0.0%) | – |
| Serial-Opt | hopper | C4.5 | 260 s | 260 s (+0.0%) | – |
| Multicore | hopper | CART | 9.13 s | 9.35 s (-2.4%) | – |
| Multicore | hopper | C4.5 | 10.1 s | 10.3 s (-2.0%) | – |
| GPU | hopper | CART | 2.82 s | 2.84 s (-0.7%) | 2.81 s (+0.1%) |
| GPU | hopper | C4.5 | 2.84 s | 2.87 s (-1.2%) | 2.84 s (-0.0%) |
| Hybrid | hopper | CART | 2.94 s | 2.98 s (-1.3%) | 2.94 s (+0.2%) |
| Hybrid | hopper | C4.5 | 2.92 s | 3.01 s (-3.0%) ! | 2.98 s (-2.0%) |

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
