### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | ampere | 1 × 1 | 4,929 s ± 0.0% | 481 W | 14 W | 504 W | 2.37 MJ ± 0.0% | 68.6 kJ | CPU socket 0 300 kJ, CPU socket 1 316 kJ, Rest of the server 1.75 MJ |
| Serial-Opt | ampere | 1 × 2 | 611 s ± 0.0% | 480 W | 13 W | 497 W | 293 kJ ± 0.0% | 8.16 kJ | CPU socket 0 40.2 kJ, CPU socket 1 36.9 kJ, Rest of the server 216 kJ |
| Parallel | ampere | 2 × 47 | 15.5 s ± 0.1% | 707 W | 241 W | 724 W | 11 kJ ± 0.4% | 3.73 kJ | CPU socket 0 2.4 kJ, CPU socket 1 2.44 kJ, Rest of the server 6.14 kJ |
| GPU | ampere | 2 × 144 | 5.05 s ± 0.1% | 615 W | 148 W | 722 W | 3.11 kJ ± 0.8% | 748 J | CPU socket 0 394 J, CPU socket 1 303 J, GPU 1.12 kJ, Rest of the server 1.29 kJ |
| Hybrid | ampere | 2 × 144 | 5.09 s ± 0.1% | 646 W | 180 W | 788 W | 3.29 kJ ± 1.3% | 915 J | CPU socket 0 475 J, CPU socket 1 306 J, GPU 1.12 kJ, Rest of the server 1.39 kJ |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | ampere | 1 × 1 | 5,855 s ± 0.0% | 483 W | 16 W | 498 W | 2.83 MJ ± 0.0% | 93.5 kJ | CPU socket 0 381 kJ, CPU socket 1 354 kJ, Rest of the server 2.09 MJ |
| Serial-Opt | ampere | 1 × 2 | 709 s ± 0.0% | 480 W | 13 W | 493 W | 340 kJ ± 0.0% | 9.5 kJ | CPU socket 0 43.1 kJ, CPU socket 1 46.3 kJ, Rest of the server 251 kJ |
| Parallel | ampere | 2 × 27 | 28.6 s ± 0.7% | 704 W | 238 W | 725 W | 20.1 kJ ± 0.7% | 6.79 kJ | CPU socket 0 4.2 kJ, CPU socket 1 4.23 kJ, Rest of the server 11.7 kJ |
| GPU | ampere | 2 × 143 | 5.09 s ± 0.0% | 620 W | 153 W | 716 W | 3.16 kJ ± 3.0% | 780 J | CPU socket 0 396 J, CPU socket 1 306 J, GPU 1.13 kJ, Rest of the server 1.33 kJ |
| Hybrid | ampere | 2 × 141 | 5.13 s ± 0.0% | 634 W | 167 W | 796 W | 3.25 kJ ± 2.7% | 857 J | CPU socket 0 476 J, CPU socket 1 309 J, GPU 1.13 kJ, Rest of the server 1.34 kJ |

### Time check

Time per tree (./tree's train time) against: the first tree of each process, the training windows the energy is summed over, and the time benchmark (cluster-ecbdl, ecbdl_15m). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | window | time benchmark |
|---|---|---|---|---|---|---|
| Serial-Simple | ampere | CART | 4,929 s | 4,929 s (+0.0%) | 4,929 s (-0.0%) | – |
| Serial-Simple | ampere | C4.5 | 5,855 s | 5,855 s (+0.0%) | 5,855 s (-0.0%) | – |
| Serial-Opt | ampere | CART | 611 s | 609 s (+0.3%) | 611 s (-0.0%) | – |
| Serial-Opt | ampere | C4.5 | 709 s | 710 s (-0.2%) | 709 s (-0.0%) | – |
| Parallel | ampere | CART | 15.5 s | 15.7 s (-1.1%) | 15.5 s (-0.0%) | – |
| Parallel | ampere | C4.5 | 28.6 s | 28.3 s (+1.0%) | 28.6 s (-0.0%) | – |
| GPU | ampere | CART | 5.05 s | 4.98 s (+1.4%) | 5.05 s (-0.0%) | 5.02 s (+0.6%) |
| GPU | ampere | C4.5 | 5.09 s | 5.05 s (+0.9%) | 5.09 s (-0.0%) | 5.05 s (+0.8%) |
| Hybrid | ampere | CART | 5.09 s | 5.04 s (+1.1%) | 5.1 s (-0.0%) | 5.07 s (+0.4%) |
| Hybrid | ampere | C4.5 | 5.13 s | 5.13 s (+0.1%) | 5.13 s (-0.0%) | 5.1 s (+0.7%) |
