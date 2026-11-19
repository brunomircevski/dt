### CART (gini), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,328 s ± 0.0% | 266 W | 130 W | 292 W | 620 kJ ± 0.0% | 302 kJ | – | CPU socket 0 216 kJ, CPU socket 1 174 kJ, Rest of the server 230 kJ |
| Serial-Opt | epyc | 3 × 1 | 147 s ± 0.0% | 271 W | 135 W | 299 W | 39.9 kJ ± 0.4% | 19.8 kJ | – | CPU socket 0 14.1 kJ, CPU socket 1 11 kJ, Rest of the server 14.7 kJ |
| Multicore | epyc | 3 × 28 | 4.54 s ± 0.6% | 594 W | 457 W | 692 W | 2.7 kJ ± 0.3% | 2.08 kJ | – | CPU socket 0 958 J, CPU socket 1 914 J, Rest of the server 827 J |
| Serial-Simple | dell | 1 × 1 | 2,589 s ± 0.0% | 307 W | 154 W | 325 W | 794 kJ ± 0.0% | 398 kJ | – | CPU socket 0 279 kJ, CPU socket 1 234 kJ, Memory 59.6 kJ, Rest of the server 221 kJ |
| Serial-Opt | dell | 3 × 1 | 243 s ± 0.5% | 309 W | 157 W | 325 W | 75.2 kJ ± 0.9% | 38.1 kJ | – | CPU socket 0 26.1 kJ, CPU socket 1 22.4 kJ, Memory 5.85 kJ, Rest of the server 20.9 kJ |
| Multicore | dell | 3 × 12 | 11.2 s ± 0.3% | 642 W | 489 W | 704 W | 7.18 kJ ± 0.2% | 5.47 kJ | – | CPU socket 0 2.47 kJ, CPU socket 1 2.45 kJ, Memory 866 J, Rest of the server 1.39 kJ |
| Serial-Simple | ampere | 1 × 1 | 4,470 s ± 0.0% | 481 W | 7 W | 521 W | 2.15 MJ ± 0.0% | 32.6 kJ | 1.12 MJ (-47.8%) | CPU socket 0 292 kJ, CPU socket 1 269 kJ, Unused GPUs 872 kJ, Rest of the server 717 kJ |
| Serial-Opt | ampere | 3 × 1 | 319 s ± 0.1% | 482 W | 9 W | 520 W | 154 kJ ± 0.2% | 2.78 kJ | 80.6 kJ (-47.6%) | CPU socket 0 21.2 kJ, CPU socket 1 19.2 kJ, Unused GPUs 62.2 kJ, Rest of the server 51.3 kJ |
| Multicore | ampere | 3 × 21 | 12.4 s ± 0.0% | 694 W | 220 W | 768 W | 8.58 kJ ± 0.3% | 2.73 kJ | 5.74 kJ (-33.1%) | CPU socket 0 1.89 kJ, CPU socket 1 1.9 kJ, Unused GPUs 2.41 kJ, Rest of the server 2.39 kJ |
| GPU | ampere | 3 × 58 | 4.29 s ± 0.1% | 685 W | 211 W | 939 W | 2.94 kJ ± 0.3% | 905 J | 2.22 kJ (-24.5%) | CPU socket 0 346 J, CPU socket 1 292 J, GPU in use 867 J, Unused GPUs 610 J, Rest of the server 821 J |
| Hybrid | ampere | 3 × 57 | 4.31 s ± 0.5% | 704 W | 231 W | 1002 W | 3.04 kJ ± 0.7% | 994 J | 2.32 kJ (-23.6%) | CPU socket 0 388 J, CPU socket 1 336 J, GPU in use 868 J, Unused GPUs 608 J, Rest of the server 836 J |
| Serial-Simple | hopper | 1 × 1 | 2,560 s ± 0.0% | 542 W | 136 W | 552 W | 1.39 MJ ± 0.0% | 348 kJ | 873 kJ (-37.0%) | CPU socket 0 346 kJ, CPU socket 1 265 kJ, Memory 98.7 kJ, Unused GPUs 520 kJ, Rest of the server 157 kJ |
| Serial-Opt | hopper | 3 × 1 | 251 s ± 0.3% | 543 W | 137 W | 551 W | 136 kJ ± 0.3% | 34.5 kJ | 86.2 kJ (-36.9%) | CPU socket 0 33.8 kJ, CPU socket 1 26 kJ, Memory 10.2 kJ, Unused GPUs 51 kJ, Rest of the server 15.4 kJ |
| Multicore | hopper | 3 × 15 | 9.4 s ± 4.1% | 899 W | 494 W | 963 W | 8.46 kJ ± 2.6% | 4.64 kJ | 6.57 kJ (-22.3%) | CPU socket 0 2.52 kJ, CPU socket 1 2.38 kJ, Memory 1.02 kJ, Unused GPUs 1.91 kJ, Rest of the server 630 J |
| GPU | hopper | 3 × 51 | 2.46 s ± 0.2% | 737 W | 331 W | 940 W | 1.81 kJ ± 0.3% | 815 J | 1.44 kJ (-20.7%) | CPU socket 0 262 J, CPU socket 1 275 J, Memory 125 J, GPU in use 628 J, Unused GPUs 381 J, Rest of the server 141 J |
| Hybrid | hopper | 3 × 51 | 2.46 s ± 0.3% | 825 W | 420 W | 1317 W | 2.03 kJ ± 0.6% | 1.03 kJ | 1.65 kJ (-18.5%) | CPU socket 0 355 J, CPU socket 1 350 J, Memory 173 J, GPU in use 627 J, Unused GPUs 381 J, Rest of the server 145 J |

### C4.5 (gain ratio), ECBDL'14 15M, depth ≤ 10, no pruning

| Implementation | Node | runs × trees | t per tree | mean P | P above idle | peak P | E per tree | E above idle | E without the unused GPUs | split of E |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,603 s ± 0.0% | 266 W | 130 W | 299 W | 693 kJ ± 0.0% | 338 kJ | – | CPU socket 0 242 kJ, CPU socket 1 194 kJ, Rest of the server 257 kJ |
| Serial-Opt | epyc | 3 × 1 | 149 s ± 0.6% | 271 W | 135 W | 299 W | 40.4 kJ ± 0.5% | 20.1 kJ | – | CPU socket 0 14.3 kJ, CPU socket 1 11.2 kJ, Rest of the server 14.9 kJ |
| Multicore | epyc | 3 × 26 | 4.77 s ± 0.5% | 595 W | 458 W | 696 W | 2.83 kJ ± 0.4% | 2.18 kJ | – | CPU socket 0 992 J, CPU socket 1 975 J, Rest of the server 868 J |
| Serial-Simple | dell | 1 × 1 | 2,867 s ± 0.0% | 308 W | 155 W | 328 W | 882 kJ ± 0.0% | 444 kJ | – | CPU socket 0 309 kJ, CPU socket 1 260 kJ, Memory 66.7 kJ, Rest of the server 245 kJ |
| Serial-Opt | dell | 3 × 1 | 254 s ± 0.5% | 309 W | 157 W | 324 W | 78.5 kJ ± 0.5% | 39.8 kJ | – | CPU socket 0 27.3 kJ, CPU socket 1 23.3 kJ, Memory 6.12 kJ, Rest of the server 21.8 kJ |
| Multicore | dell | 3 × 9 | 13.4 s ± 1.3% | 622 W | 470 W | 704 W | 8.34 kJ ± 0.9% | 6.3 kJ | – | CPU socket 0 2.88 kJ, CPU socket 1 2.85 kJ, Memory 977 J, Rest of the server 1.64 kJ |
| Serial-Simple | ampere | 1 × 1 | 5,185 s ± 0.0% | 481 W | 7 W | 538 W | 2.49 MJ ± 0.0% | 35.7 kJ | 1.3 MJ (-47.7%) | CPU socket 0 339 kJ, CPU socket 1 311 kJ, Unused GPUs 1.01 MJ, Rest of the server 832 kJ |
| Serial-Opt | ampere | 3 × 1 | 328 s ± 0.0% | 482 W | 8 W | 514 W | 158 kJ ± 0.2% | 2.71 kJ | 82.8 kJ (-47.5%) | CPU socket 0 21.8 kJ, CPU socket 1 19.7 kJ, Unused GPUs 63.8 kJ, Rest of the server 52.7 kJ |
| Multicore | ampere | 3 × 20 | 13 s ± 0.3% | 694 W | 220 W | 804 W | 9.02 kJ ± 0.2% | 2.86 kJ | 6.02 kJ (-33.3%) | CPU socket 0 1.98 kJ, CPU socket 1 1.98 kJ, Unused GPUs 2.55 kJ, Rest of the server 2.51 kJ |
| GPU | ampere | 3 × 57 | 4.34 s ± 1.0% | 684 W | 210 W | 782 W | 2.97 kJ ± 1.0% | 911 J | 2.24 kJ (-24.4%) | CPU socket 0 348 J, CPU socket 1 297 J, GPU in use 876 J, Unused GPUs 616 J, Rest of the server 830 J |
| Hybrid | ampere | 3 × 57 | 4.34 s ± 0.1% | 705 W | 231 W | 1002 W | 3.06 kJ ± 0.2% | 1 kJ | 2.34 kJ (-23.5%) | CPU socket 0 391 J, CPU socket 1 340 J, GPU in use 875 J, Unused GPUs 610 J, Rest of the server 842 J |
| Serial-Simple | hopper | 1 × 1 | 2,792 s ± 0.0% | 541 W | 136 W | 554 W | 1.51 MJ ± 0.0% | 379 kJ | 952 kJ (-37.0%) | CPU socket 0 377 kJ, CPU socket 1 289 kJ, Memory 108 kJ, Unused GPUs 567 kJ, Rest of the server 171 kJ |
| Serial-Opt | hopper | 3 × 1 | 260 s ± 0.9% | 542 W | 137 W | 551 W | 141 kJ ± 0.9% | 35.6 kJ | 89 kJ (-36.9%) | CPU socket 0 34.9 kJ, CPU socket 1 26.9 kJ, Memory 10.6 kJ, Unused GPUs 52.6 kJ, Rest of the server 16 kJ |
| Multicore | hopper | 3 × 12 | 10.7 s ± 4.3% | 882 W | 477 W | 945 W | 9.42 kJ ± 2.7% | 5.09 kJ | 7.28 kJ (-22.7%) | CPU socket 0 2.83 kJ, CPU socket 1 2.57 kJ, Memory 1.13 kJ, Unused GPUs 2.17 kJ, Rest of the server 712 J |
| GPU | hopper | 3 × 50 | 2.51 s ± 0.3% | 738 W | 333 W | 1219 W | 1.86 kJ ± 0.8% | 836 J | 1.47 kJ (-20.7%) | CPU socket 0 277 J, CPU socket 1 280 J, Memory 127 J, GPU in use 640 J, Unused GPUs 389 J, Rest of the server 143 J |
| Hybrid | hopper | 3 × 51 | 2.47 s ± 0.6% | 832 W | 426 W | 1443 W | 2.05 kJ ± 0.9% | 1.05 kJ | 1.68 kJ (-18.4%) | CPU socket 0 357 J, CPU socket 1 356 J, Memory 172 J, GPU in use 639 J, Unused GPUs 382 J, Rest of the server 146 J |

### The meter model

Each node's meter, fitted over every reading of its job: reading(t) = mean over [t − D − W, t − D] of a + b × CPU + c × GPUs + d × busy CPUs, where CPU = RAPL packages + DRAM (W), GPUs = every GPU's NVML power (W) and busy CPUs = how many CPUs are busy (/proc/stat). Halves: the same fit on the first and on the second half of the job's readings.

| Node | meter | refresh | readings | D | W | model | first half | second half | rms | idle: model / meter |
|---|---|---|---|---|---|---|---|---|---|---|
| ampere | psu_W | 59.2 s (point) | 427 | 1.0 s | 2.5 s | 92 W + 1.27·CPU + 1.18·GPUs − 0.38 W·busy CPUs | 78 W + 1.43·CPU + 1.16·GPUs − 1.47 W·busy CPUs | 87 W + 1.31·CPU + 1.18·GPUs − 0.32 W·busy CPUs | 22.8 W | 474 / 468 W |
| dell4 | sys_W | 2.1 s | 4105 | 0.4 s | 2.5 s | 62 W + 1.10·CPU + 0.17 W·busy CPUs | 63 W + 1.10·CPU + 0.20 W·busy CPUs | 61 W + 1.11·CPU + 0.16 W·busy CPUs | 17.7 W | 153 / 155 W |
| epyc1 | sys_W | 6.3 s | 960 | 1.3 s | 2.5 s | 39 W + 1.36·CPU − 0.10 W·busy CPUs | 39 W + 1.36·CPU − 0.30 W·busy CPUs | 39 W + 1.36·CPU − 0.01 W·busy CPUs | 13.2 W | 137 / 138 W |
| hopper | sys_W | 1.0 s | 15780 | 0.5 s | 1 s | 57 W + 1.03·CPU + 0.99·GPUs − 0.14 W·busy CPUs | 56 W + 1.03·CPU + 0.99·GPUs − 0.14 W·busy CPUs | 58 W + 1.03·CPU + 0.98·GPUs − 0.14 W·busy CPUs | 12.2 W | 406 / 405 W |

### PSU energy: the model against the meter

E per tree from the model (used everywhere above) against: the model with each half's coefficients; the model fitted without any reading of this case's processes (held out: a load the model has not seen) and those held-out readings' mean residual ± its standard error; the model without the busy-CPUs term; the meter's own readings over the same windows (held, shifted by D + W/2; ampere: the mean of its point readings in the windows), which is exact only for windows much longer than the meter's refresh; and the mean residual (reading − model) of the readings whose averaging window lies inside the training windows, ± its standard error, relative to the mean power.

| Implementation | Node | Algorithm | E per tree (model) | halves | held out | held-out readings: mean residual | no busy term | meter | readings inside | mean residual inside |
|---|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 620 kJ | -0.0% / -0.0% | +0.1% | 167: -0.6 ± 0.1 W | -0.1% | 619 kJ (-0.1%) | 151 | -0.6 ± 0.1 W (-0.2%) |
| Serial-Simple | epyc | C4.5 | 693 kJ | -0.0% / -0.0% | -0.0% | 184: +0.1 ± 0.1 W | -0.1% | 693 kJ (-0.0%) | 166 | -0.1 ± 0.1 W (-0.0%) |
| Serial-Opt | epyc | CART | 39.9 kJ | -0.0% / -0.0% | -0.1% | 91: +0.9 ± 0.5 W | -0.2% | 40.2 kJ (+0.7%) | 49 | +2.1 ± 0.2 W (+0.8%) |
| Serial-Opt | epyc | C4.5 | 40.4 kJ | -0.0% / -0.0% | -0.0% | 90: +0.8 ± 0.7 W | -0.2% | 40.7 kJ (+0.7%) | 47 | +2.0 ± 0.2 W (+0.7%) |
| Multicore | epyc | CART | 2.7 kJ | -1.6% / +0.5% | -0.5% | 99: +2.5 ± 2.6 W | +0.1% | 2.67 kJ (-1.0%) | 28 | +0.3 ± 2.3 W (+0.0%) |
| Multicore | epyc | C4.5 | 2.83 kJ | -1.6% / +0.5% | -0.7% | 86: +4.2 ± 2.6 W | +0.1% | 2.8 kJ (-1.1%) | 26 | +3.3 ± 1.5 W (+0.6%) |
| Serial-Simple | dell | CART | 794 kJ | +0.1% / -0.1% | -0.1% | 920: +0.6 ± 0.4 W | +0.2% | 795 kJ (+0.2%) | 850 | +0.5 ± 0.4 W (+0.2%) |
| Serial-Simple | dell | C4.5 | 882 kJ | +0.1% / -0.1% | +0.0% | 1023: -0.2 ± 0.1 W | +0.2% | 882 kJ (-0.0%) | 977 | -0.0 ± 0.0 W (-0.0%) |
| Serial-Opt | dell | CART | 75.2 kJ | +0.1% / -0.1% | +0.1% | 449: -1.6 ± 0.5 W | +0.2% | 74.9 kJ (-0.4%) | 227 | -1.3 ± 0.1 W (-0.4%) |
| Serial-Opt | dell | C4.5 | 78.5 kJ | +0.1% / -0.1% | +0.0% | 433: -0.5 ± 0.3 W | +0.2% | 78.3 kJ (-0.3%) | 221 | -0.7 ± 0.1 W (-0.2%) |
| Multicore | dell | CART | 7.18 kJ | +0.2% / -0.1% | -0.3% | 279: +4.1 ± 2.2 W | -0.2% | 7.13 kJ (-0.6%) | 144 | -4.4 ± 1.6 W (-0.7%) |
| Multicore | dell | C4.5 | 8.34 kJ | +0.2% / -0.1% | +0.3% | 230: -2.8 ± 2.8 W | -0.3% | 8.22 kJ (-1.5%) | 139 | -5.3 ± 2.6 W (-0.9%) |
| Serial-Simple | ampere | CART | 2.15 MJ | +0.1% / -0.1% | -0.1% | 80: +1.8 ± 0.3 W | -0.0% | 2.16 MJ (+0.4%) | 75 | +1.5 ± 0.3 W (+0.3%) |
| Serial-Simple | ampere | C4.5 | 2.49 MJ | +0.1% / -0.1% | -0.3% | 92: +4.0 ± 0.3 W | -0.0% | 2.51 MJ (+0.6%) | 88 | +2.9 ± 0.3 W (+0.6%) |
| Serial-Opt | ampere | CART | 154 kJ | +0.1% / -0.0% | -0.0% | 29: +0.4 ± 0.5 W | -0.0% | 154 kJ (+0.3%) | 16 | +1.2 ± 0.7 W (+0.3%) |
| Serial-Opt | ampere | C4.5 | 158 kJ | +0.1% / -0.0% | +0.0% | 28: -0.2 ± 0.7 W | -0.0% | 159 kJ (+0.5%) | 16 | +1.1 ± 0.9 W (+0.2%) |
| Multicore | ampere | CART | 8.58 kJ | -2.6% / +1.4% | -0.4% | 15: +7.3 ± 4.5 W | -0.1% | 8.65 kJ (+0.8%) | 10 | +2.5 ± 1.9 W (+0.4%) |
| Multicore | ampere | C4.5 | 9.02 kJ | -2.7% / +1.4% | -0.4% | 15: +10.6 ± 8.6 W | -0.1% | 9.09 kJ (+0.8%) | 11 | +14.2 ± 9.2 W (+2.0%) |
| GPU | ampere | CART | 2.94 kJ | -0.6% / +0.1% | +1.1% | 24: -13.1 ± 7.8 W | -0.2% | 2.73 kJ (-7.1%) | 7 | -9.8 ± 5.7 W (-1.4%) |
| GPU | ampere | C4.5 | 2.97 kJ | -0.5% / +0.1% | -0.8% | 23: +20.4 ± 6.6 W | -0.2% | 3 kJ (+1.1%) | 9 | +1.3 ± 2.4 W (+0.2%) |
| Hybrid | ampere | CART | 3.04 kJ | -1.1% / +0.3% | -0.7% | 24: +5.7 ± 10.0 W | -0.1% | 3.03 kJ (-0.3%) | 5 | +21.3 ± 24.5 W (+3.0%) |
| Hybrid | ampere | C4.5 | 3.06 kJ | -1.1% / +0.3% | +0.6% | 24: -11.4 ± 10.7 W | -0.1% | 3.01 kJ (-1.7%) | 7 | -0.6 ± 15.1 W (-0.1%) |
| Serial-Simple | hopper | CART | 1.39 MJ | -0.0% / +0.1% | +0.1% | 1430: -1.1 ± 0.0 W | -0.2% | 1.38 MJ (-0.2%) | 1349 | -0.9 ± 0.0 W (-0.2%) |
| Serial-Simple | hopper | C4.5 | 1.51 MJ | -0.0% / +0.1% | +0.1% | 1532: -1.1 ± 0.0 W | -0.2% | 1.51 MJ (-0.2%) | 1458 | -0.9 ± 0.0 W (-0.2%) |
| Serial-Opt | hopper | CART | 136 kJ | -0.0% / +0.1% | +0.0% | 636: -2.1 ± 0.1 W | -0.2% | 136 kJ (-0.5%) | 366 | -2.5 ± 0.0 W (-0.5%) |
| Serial-Opt | hopper | C4.5 | 141 kJ | -0.0% / +0.1% | +0.0% | 692: -1.5 ± 0.1 W | -0.2% | 140 kJ (-0.4%) | 424 | -2.1 ± 0.0 W (-0.4%) |
| Multicore | hopper | CART | 8.46 kJ | -0.0% / +0.0% | -0.1% | 465: +3.2 ± 0.8 W | -0.3% | 8.41 kJ (-0.5%) | 350 | +1.3 ± 0.5 W (+0.1%) |
| Multicore | hopper | C4.5 | 9.42 kJ | -0.0% / +0.0% | -0.1% | 435: +3.0 ± 0.8 W | -0.3% | 9.37 kJ (-0.5%) | 325 | +0.8 ± 0.5 W (+0.1%) |
| GPU | hopper | CART | 1.81 kJ | +0.1% / -0.0% | -0.1% | 2197: +0.8 ± 0.2 W | -0.1% | 1.83 kJ (+0.8%) | 213 | +6.8 ± 0.8 W (+0.9%) |
| GPU | hopper | C4.5 | 1.86 kJ | +0.1% / -0.0% | -0.0% | 2180: -0.4 ± 0.3 W | -0.1% | 1.87 kJ (+0.7%) | 220 | -0.1 ± 1.2 W (-0.0%) |
| Hybrid | hopper | CART | 2.03 kJ | +0.1% / -0.0% | +0.0% | 2208: -0.1 ± 0.4 W | -0.1% | 2.01 kJ (-0.7%) | 223 | -0.9 ± 2.4 W (-0.1%) |
| Hybrid | hopper | C4.5 | 2.05 kJ | +0.1% / -0.0% | +0.1% | 2218: +0.1 ± 0.4 W | -0.1% | 2.03 kJ (-0.9%) | 230 | -1.6 ± 2.9 W (-0.2%) |

### Time check

Time per tree (./tree's train time) against the first tree of each process and the time benchmark (cluster-ecbdl, ecbdl_15m; dell's Multicore ran on dell1, not comparable). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | time benchmark |
|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 2,328 s | 2,328 s (+0.0%) | 2,316 s (+0.5%) |
| Serial-Simple | epyc | C4.5 | 2,603 s | 2,603 s (+0.0%) | 2,628 s (-1.0%) |
| Serial-Opt | epyc | CART | 147 s | 147 s (+0.0%) | 148 s (-0.7%) |
| Serial-Opt | epyc | C4.5 | 149 s | 149 s (+0.0%) | 149 s (+0.4%) |
| Multicore | epyc | CART | 4.54 s | 4.48 s (+1.4%) | 4.47 s (+1.7%) |
| Multicore | epyc | C4.5 | 4.77 s | 4.69 s (+1.5%) | 4.73 s (+0.7%) |
| Serial-Simple | dell | CART | 2,589 s | 2,589 s (+0.0%) | 2,541 s (+1.9%) |
| Serial-Simple | dell | C4.5 | 2,867 s | 2,867 s (+0.0%) | 2,820 s (+1.7%) |
| Serial-Opt | dell | CART | 243 s | 243 s (+0.0%) | 252 s (-3.6%) ! |
| Serial-Opt | dell | C4.5 | 254 s | 254 s (+0.0%) | 261 s (-2.9%) |
| Multicore | dell | CART | 11.2 s | 11.2 s (-0.2%) | – |
| Multicore | dell | C4.5 | 13.4 s | 13.4 s (-0.3%) | – |
| Serial-Simple | ampere | CART | 4,470 s | 4,470 s (+0.0%) | – |
| Serial-Simple | ampere | C4.5 | 5,185 s | 5,185 s (+0.0%) | – |
| Serial-Opt | ampere | CART | 319 s | 319 s (+0.0%) | – |
| Serial-Opt | ampere | C4.5 | 328 s | 328 s (+0.0%) | – |
| Multicore | ampere | CART | 12.4 s | 12.6 s (-1.7%) | – |
| Multicore | ampere | C4.5 | 13 s | 13 s (-0.2%) | – |
| GPU | ampere | CART | 4.29 s | 4.26 s (+0.5%) | 4.23 s (+1.3%) |
| GPU | ampere | C4.5 | 4.34 s | 4.32 s (+0.4%) | 4.35 s (-0.2%) |
| Hybrid | ampere | CART | 4.31 s | 4.38 s (-1.6%) | 4.22 s (+2.2%) |
| Hybrid | ampere | C4.5 | 4.34 s | 4.31 s (+0.5%) | 4.26 s (+1.8%) |
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
| ampere | GPU | 0 s | 12 | 4.922 s | 5.480 s | 5.570 s |
| ampere | GPU | 2 s | 12 | 4.223 s | 4.323 s | 4.434 s |
| ampere | GPU | 5 s | 12 | 4.405 s | 4.332 s | 4.419 s |
| ampere | GPU | 15 s | 12 | 4.371 s | 4.367 s | 4.430 s |
| ampere | Hybrid | 0 s | 12 | 4.690 s | 5.144 s | 5.309 s |
| ampere | Hybrid | 2 s | 12 | 4.293 s | 4.336 s | 4.425 s |
| ampere | Hybrid | 5 s | 12 | 4.295 s | 4.337 s | 4.466 s |
| ampere | Hybrid | 15 s | 12 | 4.355 s | 4.352 s | 4.405 s |
| hopper | GPU | 0 s | 12 | 2.598 s | 2.766 s | 2.847 s |
| hopper | GPU | 2 s | 12 | 2.526 s | 2.700 s | 2.844 s |
| hopper | GPU | 5 s | 12 | 2.544 s | 2.477 s | 2.551 s |
| hopper | GPU | 15 s | 12 | 2.452 s | 2.462 s | 2.495 s |
| hopper | Hybrid | 0 s | 12 | 2.694 s | 2.814 s | 2.840 s |
| hopper | Hybrid | 2 s | 12 | 2.547 s | 2.570 s | 2.740 s |
| hopper | Hybrid | 5 s | 12 | 2.478 s | 2.450 s | 2.488 s |
| hopper | Hybrid | 15 s | 12 | 2.485 s | 2.468 s | 2.489 s |
