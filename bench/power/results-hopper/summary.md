### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | hopper | 1 × 1 | 2,520 s ± 0.0% | 540 W | 137 W | 552 W | 1.36 MJ ± 0.0% | 346 kJ | CPU socket 0 340 kJ, CPU socket 1 259 kJ, Rest of the server 761 kJ |
| Serial-Opt | hopper | 1 × 1 | 341 s ± 0.0% | 541 W | 139 W | 544 W | 185 kJ ± 0.0% | 47.3 kJ | CPU socket 0 40.4 kJ, CPU socket 1 40.6 kJ, Rest of the server 104 kJ |
| Parallel | hopper | 3 × 11 | 12.4 s ± 0.3% | 901 W | 499 W | 933 W | 11.2 kJ ± 0.3% | 6.18 kJ | CPU socket 0 3.33 kJ, CPU socket 1 3.29 kJ, Rest of the server 4.56 kJ |
| GPU | hopper | 3 × 42 | 2.99 s ± 0.1% | 718 W | 316 W | 808 W | 2.15 kJ ± 0.4% | 945 J | CPU socket 0 281 J, CPU socket 1 323 J, GPU 850 J, Rest of the server 697 J |
| Hybrid | hopper | 3 × 42 | 3.01 s ± 0.4% | 790 W | 387 W | 937 W | 2.38 kJ ± 0.6% | 1.16 kJ | CPU socket 0 328 J, CPU socket 1 454 J, GPU 854 J, Rest of the server 739 J |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | hopper | 1 × 1 | 2,757 s ± 0.0% | 540 W | 137 W | 546 W | 1.49 MJ ± 0.0% | 378 kJ | CPU socket 0 371 kJ, CPU socket 1 284 kJ, Rest of the server 833 kJ |
| Serial-Opt | hopper | 1 × 1 | 415 s ± 0.0% | 538 W | 135 W | 543 W | 223 kJ ± 0.0% | 56 kJ | CPU socket 0 49.2 kJ, CPU socket 1 49.3 kJ, Rest of the server 124 kJ |
| Parallel | hopper | 3 × 10 | 14.9 s ± 0.1% | 898 W | 496 W | 956 W | 13.4 kJ ± 0.1% | 7.4 kJ | CPU socket 0 3.99 kJ, CPU socket 1 3.94 kJ, Rest of the server 5.48 kJ |
| GPU | hopper | 3 × 42 | 3.03 s ± 0.3% | 720 W | 317 W | 820 W | 2.18 kJ ± 0.3% | 960 J | CPU socket 0 286 J, CPU socket 1 326 J, GPU 865 J, Rest of the server 702 J |
| Hybrid | hopper | 3 × 41 | 3.02 s ± 0.4% | 797 W | 394 W | 940 W | 2.41 kJ ± 0.4% | 1.19 kJ | CPU socket 0 336 J, CPU socket 1 461 J, GPU 863 J, Rest of the server 746 J |

### Time check

Time per tree (./tree's train time) against: the first tree of each process, the training windows the energy is summed over, and the time benchmark (cluster-ecbdl, ecbdl_15m). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | window | time benchmark |
|---|---|---|---|---|---|---|
| Serial-Simple | hopper | CART | 2,520 s | 2,520 s (+0.0%) | 2,520 s (-0.0%) | – |
| Serial-Simple | hopper | C4.5 | 2,757 s | 2,757 s (+0.0%) | 2,757 s (-0.0%) | – |
| Serial-Opt | hopper | CART | 341 s | 341 s (+0.0%) | 341 s (-0.0%) | – |
| Serial-Opt | hopper | C4.5 | 415 s | 415 s (+0.0%) | 415 s (-0.0%) | – |
| Parallel | hopper | CART | 12.4 s | 12.4 s (-0.3%) | 12.4 s (-0.0%) | – |
| Parallel | hopper | C4.5 | 14.9 s | 14.9 s (-0.0%) | 14.9 s (-0.0%) | – |
| GPU | hopper | CART | 2.99 s | 3 s (-0.3%) | 2.99 s (-0.0%) | 2.97 s (+0.9%) |
| GPU | hopper | C4.5 | 3.03 s | 3.01 s (+0.7%) | 3.03 s (-0.0%) | 2.99 s (+1.2%) |
| Hybrid | hopper | CART | 3.01 s | 3.01 s (-0.2%) | 3.01 s (-0.0%) | 2.97 s (+1.4%) |
| Hybrid | hopper | C4.5 | 3.02 s | 3.02 s (-0.1%) | 3.02 s (-0.0%) | 2.99 s (+1.0%) |
