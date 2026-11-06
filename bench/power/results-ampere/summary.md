### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | ampere | 1 × 1 | 4,931 s ± 0.0% | 480 W | 14 W | 491 W | 2.37 MJ ± 0.0% | 69 kJ | CPU socket 0 305 kJ, CPU socket 1 313 kJ, Rest of the server 1.75 MJ |
| Serial-Opt | ampere | 1 × 2 | 623 s ± 0.0% | 478 W | 12 W | 487 W | 298 kJ ± 0.0% | 7.55 kJ | CPU socket 0 38 kJ, CPU socket 1 40.6 kJ, Rest of the server 220 kJ |
| Parallel | ampere | 2 × 47 | 15.4 s ± 0.0% | 700 W | 234 W | 716 W | 11 kJ ± 0.6% | 3.67 kJ | CPU socket 0 2.42 kJ, CPU socket 1 2.44 kJ, Rest of the server 6.13 kJ |
| GPU | ampere | 2 × 103 | 7.09 s ± 0.1% | 650 W | 184 W | 725 W | 4.67 kJ ± 0.7% | 1.32 kJ | CPU socket 0 498 J, CPU socket 1 408 J, GPU 1.32 kJ, Rest of the server 2.44 kJ |
| Hybrid | ampere | 2 × 105 | 6.9 s ± 0.1% | 663 W | 197 W | 800 W | 4.8 kJ ± 5.0% | 1.43 kJ | CPU socket 0 585 J, CPU socket 1 415 J, GPU 1.32 kJ, Rest of the server 2.48 kJ |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | ampere | 1 × 1 | 5,706 s ± 0.0% | 481 W | 15 W | 486 W | 2.75 MJ ± 0.0% | 85.1 kJ | CPU socket 0 370 kJ, CPU socket 1 343 kJ, Rest of the server 2.03 MJ |
| Serial-Opt | ampere | 1 × 2 | 720 s ± 0.0% | 481 W | 15 W | 510 W | 347 kJ ± 0.0% | 10.7 kJ | CPU socket 0 43.8 kJ, CPU socket 1 46.9 kJ, Rest of the server 256 kJ |
| Parallel | ampere | 2 × 26 | 28 s ± 0.1% | 692 W | 225 W | 731 W | 19.6 kJ ± 0.0% | 6.38 kJ | CPU socket 0 4.18 kJ, CPU socket 1 4.18 kJ, Rest of the server 11.2 kJ |
| GPU | ampere | 2 × 102 | 7.12 s ± 0.1% | 638 W | 171 W | 718 W | 4.6 kJ ± 0.2% | 1.24 kJ | CPU socket 0 500 J, CPU socket 1 411 J, GPU 1.33 kJ, Rest of the server 2.36 kJ |
| Hybrid | ampere | 2 × 105 | 6.93 s ± 0.3% | 649 W | 183 W | 790 W | 4.72 kJ ± 5.8% | 1.33 kJ | CPU socket 0 585 J, CPU socket 1 417 J, GPU 1.33 kJ, Rest of the server 2.39 kJ |
