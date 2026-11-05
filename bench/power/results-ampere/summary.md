### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | ampere | 1 × 1 | 4,941 s ± 0.0% | 125 W | 7 W | 153 W | 618 kJ ± 0.0% | 35.6 kJ | CPU socket 0 300 kJ, CPU socket 1 318 kJ |
| Serial-Opt | ampere | 3 × 1 | 636 s ± 4.0% | 126 W | 9 W | 183 W | 80.4 kJ ± 5.0% | 5.45 kJ | CPU socket 0 40.1 kJ, CPU socket 1 40.3 kJ |
| Parallel | ampere | 5 × 1st of 2 | 15.4 s ± 0.3% | 307 W | 189 W | 320 W | 4.82 kJ ± 0.2% | 2.97 kJ | CPU socket 0 2.4 kJ, CPU socket 1 2.42 kJ |
| GPU | ampere | 5 × 1st of 5 | 5.01 s ± 0.3% | 354 W | 182 W | 462 W | 1.8 kJ ± 0.7% | 924 J | CPU socket 0 395 J, CPU socket 1 304 J, GPU 1.1 kJ |
| Hybrid | ampere | 5 × 1st of 5 | 5.09 s ± 0.3% | 358 W | 186 W | 512 W | 1.94 kJ ± 0.4% | 1.01 kJ | CPU socket 0 497 J, CPU socket 1 325 J, GPU 1.12 kJ |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | ampere | 1 × 1 | 5,656 s ± 0.0% | 125 W | 7 W | 173 W | 706 kJ ± 0.0% | 39.2 kJ | CPU socket 0 366 kJ, CPU socket 1 339 kJ |
| Serial-Opt | ampere | 3 × 1 | 717 s ± 0.3% | 125 W | 8 W | 157 W | 90 kJ ± 0.2% | 5.43 kJ | CPU socket 0 45.7 kJ, CPU socket 1 44.3 kJ |
| Parallel | ampere | 5 × 1st of 2 | 27.7 s ± 0.3% | 293 W | 175 W | 320 W | 8.22 kJ ± 0.3% | 4.92 kJ | CPU socket 0 4.11 kJ, CPU socket 1 4.12 kJ |
| GPU | ampere | 5 × 1st of 5 | 5.06 s ± 0.5% | 352 W | 180 W | 464 W | 1.81 kJ ± 1.5% | 926 J | CPU socket 0 400 J, CPU socket 1 308 J, GPU 1.1 kJ |
| Hybrid | ampere | 5 × 1st of 5 | 5.12 s ± 0.3% | 358 W | 186 W | 521 W | 1.95 kJ ± 0.6% | 1.01 kJ | CPU socket 0 499 J, CPU socket 1 327 J, GPU 1.13 kJ |

### Later trees in the same process (ampere)

| Implementation | Algorithm | first tree: t, E | later trees: t, E (mean) |
|---|---|---|---|
| Parallel | CART | 15.4 s, 4.82 kJ | 15.4 s, 4.83 kJ |
| Parallel | C4.5 | 27.7 s, 8.22 kJ | 27.9 s, 8.29 kJ |
| GPU | CART | 5.01 s, 1.8 kJ | 7.13 s, 2.2 kJ |
| GPU | C4.5 | 5.06 s, 1.81 kJ | 7.17 s, 2.22 kJ |
| Hybrid | CART | 5.09 s, 1.94 kJ | 6.92 s, 2.28 kJ |
| Hybrid | C4.5 | 5.12 s, 1.95 kJ | 6.95 s, 2.29 kJ |
