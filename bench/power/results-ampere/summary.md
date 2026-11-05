### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | ampere | 1 × 1 | 4,941 s ± 0.0% | 319 W | 5 W | 416 W | 1.58 MJ ± 0.0% | 23.5 kJ | CPU socket 0 300 kJ, CPU socket 1 318 kJ, GPU 0, in use 259 kJ, GPUs 1–3, unused 699 kJ |
| Serial-Opt | ampere | 3 × 1 | 636 s ± 4.0% | 321 W | 6 W | 435 W | 204 kJ ± 4.4% | 4.13 kJ | CPU socket 0 40.1 kJ, CPU socket 1 40.3 kJ, GPU 0, in use 33.5 kJ, GPUs 1–3, unused 90 kJ |
| Parallel | ampere | 5 × 1st of 2 | 15.4 s ± 0.3% | 502 W | 187 W | 515 W | 7.89 kJ ± 0.4% | 2.95 kJ | CPU socket 0 2.4 kJ, CPU socket 1 2.42 kJ, GPU 0, in use 838 J, GPUs 1–3, unused 2.22 kJ |
| GPU | ampere | 5 × 1st of 5 | 5.01 s ± 0.3% | 496 W | 181 W | 604 W | 2.52 kJ ± 0.8% | 922 J | CPU socket 0 395 J, CPU socket 1 304 J, GPU 0, in use 1.1 kJ, GPUs 1–3, unused 718 J |
| Hybrid | ampere | 5 × 1st of 5 | 5.09 s ± 0.3% | 500 W | 186 W | 654 W | 2.71 kJ ± 0.3% | 1.01 kJ | CPU socket 0 497 J, CPU socket 1 325 J, GPU 0, in use 1.12 kJ, GPUs 1–3, unused 769 J |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | ampere | 1 × 1 | 5,656 s ± 0.0% | 319 W | 5 W | 455 W | 1.8 MJ ± 0.0% | 26.1 kJ | CPU socket 0 366 kJ, CPU socket 1 339 kJ, GPU 0, in use 297 kJ, GPUs 1–3, unused 800 kJ |
| Serial-Opt | ampere | 3 × 1 | 717 s ± 0.3% | 320 W | 5 W | 453 W | 229 kJ ± 0.4% | 3.85 kJ | CPU socket 0 45.7 kJ, CPU socket 1 44.3 kJ, GPU 0, in use 37.8 kJ, GPUs 1–3, unused 102 kJ |
| Parallel | ampere | 5 × 1st of 2 | 27.7 s ± 0.3% | 488 W | 173 W | 515 W | 13.7 kJ ± 0.4% | 4.87 kJ | CPU socket 0 4.11 kJ, CPU socket 1 4.12 kJ, GPU 0, in use 1.49 kJ, GPUs 1–3, unused 3.96 kJ |
| GPU | ampere | 5 × 1st of 5 | 5.06 s ± 0.5% | 494 W | 180 W | 606 W | 2.54 kJ ± 1.4% | 928 J | CPU socket 0 400 J, CPU socket 1 308 J, GPU 0, in use 1.1 kJ, GPUs 1–3, unused 731 J |
| Hybrid | ampere | 5 × 1st of 5 | 5.12 s ± 0.3% | 500 W | 185 W | 663 W | 2.72 kJ ± 0.5% | 1.01 kJ | CPU socket 0 499 J, CPU socket 1 327 J, GPU 0, in use 1.13 kJ, GPUs 1–3, unused 770 J |

### Later trees in the same process (ampere)

| Implementation | Algorithm | first tree: t, E | later trees: t, E (mean) |
|---|---|---|---|
| Parallel | CART | 15.4 s, 7.89 kJ | 15.4 s, 7.89 kJ |
| Parallel | C4.5 | 27.7 s, 13.7 kJ | 27.9 s, 13.8 kJ |
| GPU | CART | 5.01 s, 2.52 kJ | 7.13 s, 3.22 kJ |
| GPU | C4.5 | 5.06 s, 2.54 kJ | 7.17 s, 3.25 kJ |
| Hybrid | CART | 5.09 s, 2.71 kJ | 6.92 s, 3.3 kJ |
| Hybrid | C4.5 | 5.12 s, 2.72 kJ | 6.95 s, 3.32 kJ |
