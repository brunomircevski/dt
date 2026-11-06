### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | hopper | 1 × 1 | 2,505 s ± 0.0% | 536 W | 135 W | 542 W | 1.34 MJ ± 0.0% | 338 kJ | CPU socket 0 296 kJ, CPU socket 1 297 kJ, Rest of the server 750 kJ |
| Serial-Opt | hopper | 1 × 1 | 337 s ± 0.0% | 541 W | 141 W | 547 W | 182 kJ ± 0.0% | 47.3 kJ | CPU socket 0 45.6 kJ, CPU socket 1 34.7 kJ, Rest of the server 102 kJ |
| Parallel | hopper | 3 × 10 | 12.2 s ± 0.2% | 900 W | 499 W | 934 W | 11.2 kJ ± 0.2% | 6.21 kJ | CPU socket 0 3.33 kJ, CPU socket 1 3.28 kJ, Rest of the server 4.59 kJ |
| GPU | hopper | 3 × 37 | 3.61 s ± 0.2% | 723 W | 322 W | 778 W | 2.64 kJ ± 0.2% | 1.18 kJ | CPU socket 0 332 J, CPU socket 1 345 J, GPU 968 J, Rest of the server 995 J |
| Hybrid | hopper | 3 × 40 | 3.24 s ± 0.1% | 787 W | 387 W | 924 W | 2.74 kJ ± 0.3% | 1.35 kJ | CPU socket 0 365 J, CPU socket 1 460 J, GPU 942 J, Rest of the server 978 J |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | hopper | 1 × 1 | 2,745 s ± 0.0% | 536 W | 135 W | 544 W | 1.47 MJ ± 0.0% | 370 kJ | CPU socket 0 323 kJ, CPU socket 1 324 kJ, Rest of the server 823 kJ |
| Serial-Opt | hopper | 1 × 1 | 401 s ± 0.0% | 541 W | 140 W | 545 W | 217 kJ ± 0.0% | 56.4 kJ | CPU socket 0 54.4 kJ, CPU socket 1 41.4 kJ, Rest of the server 122 kJ |
| Parallel | hopper | 3 × 9 | 15 s ± 0.3% | 900 W | 499 W | 951 W | 13.7 kJ ± 0.3% | 7.58 kJ | CPU socket 0 4.04 kJ, CPU socket 1 3.99 kJ, Rest of the server 5.63 kJ |
| GPU | hopper | 3 × 37 | 3.64 s ± 0.4% | 718 W | 317 W | 775 W | 2.64 kJ ± 0.2% | 1.17 kJ | CPU socket 0 334 J, CPU socket 1 347 J, GPU 973 J, Rest of the server 987 J |
| Hybrid | hopper | 3 × 40 | 3.27 s ± 0.5% | 786 W | 385 W | 925 W | 2.76 kJ ± 0.3% | 1.35 kJ | CPU socket 0 368 J, CPU socket 1 463 J, GPU 948 J, Rest of the server 981 J |
