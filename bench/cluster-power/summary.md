### CART (gini), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,328 s ± 0.0% | 266 W | 128 W | 291 W | 619 kJ ± 0.0% | 299 kJ | – | CPU socket 0 216 kJ, CPU socket 1 174 kJ, Rest of the server 229 kJ |
| Serial-Opt | epyc | 3 × 1 | 147 s ± 0.0% | 271 W | 134 W | 298 W | 39.9 kJ ± 0.4% | 19.6 kJ | – | CPU socket 0 14.1 kJ, CPU socket 1 11 kJ, Rest of the server 14.7 kJ |
| Multicore | epyc | 3 × 28 | 4.54 s ± 0.6% | 594 W | 457 W | 691 W | 2.7 kJ ± 0.3% | 2.08 kJ | – | CPU socket 0 958 J, CPU socket 1 914 J, Rest of the server 829 J |
| Serial-Simple | hopper | 1 × 1 | 2,560 s ± 0.0% | 540 W | 134 W | 547 W | 1.38 MJ ± 0.0% | 344 kJ | 863 kJ (-37.6%) | CPU socket 0 346 kJ, CPU socket 1 265 kJ, Memory 98.7 kJ, Unused GPUs 520 kJ, Rest of the server 153 kJ |
| Serial-Opt | hopper | 3 × 1 | 251 s ± 0.3% | 542 W | 136 W | 549 W | 136 kJ ± 0.3% | 34.1 kJ | 85.2 kJ (-37.4%) | CPU socket 0 33.8 kJ, CPU socket 1 26 kJ, Memory 10.2 kJ, Unused GPUs 51 kJ, Rest of the server 15.2 kJ |
| Multicore | hopper | 3 × 15 | 9.4 s ± 4.1% | 897 W | 491 W | 955 W | 8.43 kJ ± 2.6% | 4.61 kJ | 6.52 kJ (-22.7%) | CPU socket 0 2.52 kJ, CPU socket 1 2.38 kJ, Memory 1.02 kJ, Unused GPUs 1.91 kJ, Rest of the server 606 J |
| GPU | hopper | 3 × 51 | 2.46 s ± 0.2% | 738 W | 332 W | 848 W | 1.81 kJ ± 0.3% | 816 J | 1.43 kJ (-21.0%) | CPU socket 0 262 J, CPU socket 1 275 J, Memory 125 J, GPU in use 628 J, Unused GPUs 381 J, Rest of the server 144 J |
| Hybrid | hopper | 3 × 51 | 2.46 s ± 0.3% | 826 W | 420 W | 1137 W | 2.03 kJ ± 0.6% | 1.03 kJ | 1.65 kJ (-18.7%) | CPU socket 0 355 J, CPU socket 1 350 J, Memory 173 J, GPU in use 627 J, Unused GPUs 381 J, Rest of the server 147 J |

### C4.5 (gain ratio), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,603 s ± 0.0% | 266 W | 129 W | 298 W | 692 kJ ± 0.0% | 335 kJ | – | CPU socket 0 242 kJ, CPU socket 1 194 kJ, Rest of the server 256 kJ |
| Serial-Opt | epyc | 3 × 1 | 149 s ± 0.6% | 271 W | 133 W | 298 W | 40.4 kJ ± 0.5% | 19.9 kJ | – | CPU socket 0 14.3 kJ, CPU socket 1 11.2 kJ, Rest of the server 14.9 kJ |
| Multicore | epyc | 3 × 26 | 4.77 s ± 0.5% | 595 W | 458 W | 695 W | 2.84 kJ ± 0.4% | 2.18 kJ | – | CPU socket 0 992 J, CPU socket 1 975 J, Rest of the server 871 J |
| Serial-Simple | hopper | 1 × 1 | 2,792 s ± 0.0% | 540 W | 134 W | 546 W | 1.51 MJ ± 0.0% | 374 kJ | 941 kJ (-37.6%) | CPU socket 0 377 kJ, CPU socket 1 289 kJ, Memory 108 kJ, Unused GPUs 567 kJ, Rest of the server 167 kJ |
| Serial-Opt | hopper | 3 × 1 | 260 s ± 0.9% | 541 W | 135 W | 548 W | 141 kJ ± 0.9% | 35.2 kJ | 88 kJ (-37.4%) | CPU socket 0 34.9 kJ, CPU socket 1 26.9 kJ, Memory 10.6 kJ, Unused GPUs 52.6 kJ, Rest of the server 15.7 kJ |
| Multicore | hopper | 3 × 12 | 10.7 s ± 4.3% | 880 W | 474 W | 942 W | 9.39 kJ ± 2.7% | 5.05 kJ | 7.22 kJ (-23.1%) | CPU socket 0 2.83 kJ, CPU socket 1 2.57 kJ, Memory 1.13 kJ, Unused GPUs 2.17 kJ, Rest of the server 684 J |
| GPU | hopper | 3 × 50 | 2.51 s ± 0.3% | 739 W | 333 W | 971 W | 1.86 kJ ± 0.8% | 839 J | 1.47 kJ (-20.9%) | CPU socket 0 277 J, CPU socket 1 280 J, Memory 127 J, GPU in use 640 J, Unused GPUs 389 J, Rest of the server 147 J |
| Hybrid | hopper | 3 × 51 | 2.47 s ± 0.6% | 832 W | 426 W | 1201 W | 2.05 kJ ± 0.9% | 1.05 kJ | 1.67 kJ (-18.6%) | CPU socket 0 357 J, CPU socket 1 356 J, Memory 172 J, GPU in use 639 J, Unused GPUs 382 J, Rest of the server 148 J |

### The meter model

Each node's meter, fitted over every reading of its job: reading(t) = mean over [t − D − W, t − D] of a + G_idle + b × CPU + GPU, where CPU = RAPL packages + DRAM (W), GPU = the NVML power of the GPU in use (W, counted once) and G_idle = the mean NVML power of the node's other GPUs over the job (measured, not fitted; none on epyc). Halves: the same fit on the first and on the second half of the job's readings.

| Node | meter | refresh | readings | D | W | model | first half | second half | rms | idle: model / meter |
|---|---|---|---|---|---|---|---|---|---|---|
| epyc1 | sys_W | 6.3 s | 960 | 1.3 s | 2.5 s | 41 W + 1.34·CPU | 44 W + 1.32·CPU | 40 W + 1.35·CPU | 13.2 W | 137 / 138 W |
| hopper | sys_W | 1.0 s | 15780 | 0.5 s | 1 s | 56 W + 154 W + 1.01·CPU + GPU | 57 W + 154 W + 1.01·CPU + GPU | 55 W + 154 W + 1.02·CPU + GPU | 12.3 W | 406 / 405 W |

### PSU energy: the model against the meter

E per tree from the model (used everywhere above) against: the model with each half's coefficients; the model fitted without any reading of this case's processes (held out: a load the model has not seen) and those held-out readings' mean residual ± its standard error; the meter's own readings over the same windows (held, shifted by D + W/2), which is exact only for windows much longer than the meter's refresh; and the mean residual (reading − model) of the readings whose averaging window lies inside the training windows, ± its standard error, relative to the mean power.

| Implementation | Node | Algorithm | E per tree (model) | halves | held out | held-out readings: mean residual | meter | readings inside | mean residual inside |
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 619 kJ | -0.3% / +0.1% | +0.0% | 167: -0.1 ± 0.1 W | 619 kJ (-0.0%) | 151 | -0.2 ± 0.1 W (-0.1%) |
| Serial-Simple | epyc | C4.5 | 692 kJ | -0.3% / +0.1% | -0.0% | 184: +0.5 ± 0.1 W | 693 kJ (+0.1%) | 166 | +0.3 ± 0.1 W (+0.1%) |
| Serial-Opt | epyc | CART | 39.9 kJ | -0.3% / +0.1% | -0.0% | 91: +1.3 ± 0.5 W | 40.2 kJ (+0.8%) | 49 | +2.5 ± 0.2 W (+0.9%) |
| Serial-Opt | epyc | C4.5 | 40.4 kJ | -0.3% / +0.1% | -0.0% | 90: +1.3 ± 0.7 W | 40.7 kJ (+0.8%) | 47 | +2.4 ± 0.2 W (+0.9%) |
| Multicore | epyc | CART | 2.7 kJ | -1.2% / +0.5% | -0.4% | 99: +1.9 ± 2.6 W | 2.67 kJ (-1.1%) | 28 | -0.3 ± 2.3 W (-0.1%) |
| Multicore | epyc | C4.5 | 2.84 kJ | -1.2% / +0.5% | -0.5% | 86: +3.3 ± 2.6 W | 2.8 kJ (-1.2%) | 26 | +2.7 ± 1.5 W (+0.5%) |
| Serial-Simple | hopper | CART | 1.38 MJ | +0.0% / -0.1% | -0.0% | 1430: +0.7 ± 0.0 W | 1.38 MJ (+0.1%) | 1349 | +0.6 ± 0.0 W (+0.1%) |
| Serial-Simple | hopper | C4.5 | 1.51 MJ | +0.0% / -0.1% | -0.0% | 1532: +0.8 ± 0.0 W | 1.51 MJ (+0.1%) | 1458 | +0.5 ± 0.0 W (+0.1%) |
| Serial-Opt | hopper | CART | 136 kJ | +0.0% / -0.1% | +0.0% | 636: -0.9 ± 0.1 W | 136 kJ (-0.3%) | 366 | -1.4 ± 0.0 W (-0.3%) |
| Serial-Opt | hopper | C4.5 | 141 kJ | +0.0% / -0.1% | +0.0% | 692: -0.5 ± 0.1 W | 140 kJ (-0.2%) | 424 | -1.1 ± 0.0 W (-0.2%) |
| Multicore | hopper | CART | 8.43 kJ | -0.1% / +0.1% | -0.1% | 465: +5.6 ± 0.8 W | 8.41 kJ (-0.2%) | 350 | +4.0 ± 0.5 W (+0.4%) |
| Multicore | hopper | C4.5 | 9.39 kJ | -0.1% / +0.1% | -0.1% | 435: +5.3 ± 0.8 W | 9.37 kJ (-0.2%) | 325 | +3.4 ± 0.5 W (+0.4%) |
| GPU | hopper | CART | 1.81 kJ | +0.0% / -0.1% | -0.0% | 2197: -0.2 ± 0.2 W | 1.83 kJ (+0.7%) | 213 | +5.4 ± 0.8 W (+0.7%) |
| GPU | hopper | C4.5 | 1.86 kJ | +0.0% / -0.1% | +0.0% | 2180: -1.5 ± 0.3 W | 1.87 kJ (+0.5%) | 220 | -1.6 ± 1.2 W (-0.2%) |
| Hybrid | hopper | CART | 2.03 kJ | +0.0% / -0.0% | +0.0% | 2208: -1.1 ± 0.4 W | 2.01 kJ (-0.8%) | 223 | -1.6 ± 2.4 W (-0.2%) |
| Hybrid | hopper | C4.5 | 2.05 kJ | +0.0% / -0.0% | +0.0% | 2218: -0.7 ± 0.4 W | 2.03 kJ (-0.9%) | 230 | -2.4 ± 2.9 W (-0.3%) |

### Time check

Time per tree (./tree's train time) against the first tree of each process and the time benchmark (cluster-ecbdl, ecbdl_15m). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 2,328 s | 2,328 s (+0.0%) | 2,316 s (+0.5%) |
| Serial-Simple | epyc | C4.5 | 2,603 s | 2,603 s (+0.0%) | 2,628 s (-1.0%) |
| Serial-Opt | epyc | CART | 147 s | 147 s (+0.0%) | 148 s (-0.7%) |
| Serial-Opt | epyc | C4.5 | 149 s | 149 s (+0.0%) | 149 s (+0.4%) |
| Multicore | epyc | CART | 4.54 s | 4.48 s (+1.4%) | 4.47 s (+1.7%) |
| Multicore | epyc | C4.5 | 4.77 s | 4.69 s (+1.5%) | 4.73 s (+0.7%) |
| Serial-Simple | hopper | CART | 2,560 s | 2,560 s (+0.0%) | – |
| Serial-Simple | hopper | C4.5 | 2,792 s | 2,792 s (+0.0%) | – |
| Serial-Opt | hopper | CART | 251 s | 251 s (+0.0%) | – |
| Serial-Opt | hopper | C4.5 | 260 s | 260 s (+0.0%) | – |
| Multicore | hopper | CART | 9.4 s | 9.75 s (-3.6%) ! | – |
| Multicore | hopper | C4.5 | 10.7 s | 11.2 s (-4.9%) ! | – |
| GPU | hopper | CART | 2.46 s | 2.45 s (+0.3%) | 2.47 s (-0.6%) |
| GPU | hopper | C4.5 | 2.51 s | 2.52 s (-0.3%) | 2.59 s (-3.1%) ! |
| Hybrid | hopper | CART | 2.46 s | 2.47 s (-0.6%) | 2.42 s (+1.8%) |
| Hybrid | hopper | C4.5 | 2.47 s | 2.49 s (-0.7%) | 2.43 s (+1.4%) |

### Warm-up check

GPU and Hybrid, CART, one process per warm-up (--repeat-warmup S: each tree's child keeps its CPU threads busy S s before its timer). Mean of trees 1–2 and of the rest.

| Node | Implementation | warm-up | trees | trees 1–2 | trees 3– | slowest |
|---|---|---|---|---|---|---|
| hopper | GPU | 0 s | 12 | 2.598 s | 2.766 s | 2.847 s |
| hopper | GPU | 2 s | 12 | 2.526 s | 2.700 s | 2.844 s |
| hopper | GPU | 5 s | 12 | 2.544 s | 2.477 s | 2.551 s |
| hopper | GPU | 15 s | 12 | 2.452 s | 2.462 s | 2.495 s |
| hopper | Hybrid | 0 s | 12 | 2.694 s | 2.814 s | 2.840 s |
| hopper | Hybrid | 2 s | 12 | 2.547 s | 2.570 s | 2.740 s |
| hopper | Hybrid | 5 s | 12 | 2.478 s | 2.450 s | 2.488 s |
| hopper | Hybrid | 15 s | 12 | 2.485 s | 2.468 s | 2.489 s |
