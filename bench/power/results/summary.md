### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 3 × 1 | 2,309 s ± 0.2% | 257 W | 147 W | 262 W | 593 kJ ± 0.1% | 339 kJ | Whole server 593 kJ |
| Serial-Opt | epyc | 3 × 1 | 225 s ± 0.2% | 262 W | 152 W | 276 W | 59 kJ ± 0.1% | 34.2 kJ | Whole server 59 kJ |
| Parallel | epyc | 3 × 18 | 6.3 s ± 0.2% | 592 W | 482 W | 610 W | 3.8 kJ ± 0.6% | 3.09 kJ | Whole server 3.8 kJ |
| Serial-Simple | dell | 3 × 1 | 2,556 s ± 0.3% | 307 W | 149 W | 313 W | 786 kJ ± 0.2% | 380 kJ | Whole server 786 kJ |
| Serial-Opt | dell | 3 × 1 | 356 s ± 1.1% | 308 W | 150 W | 367 W | 110 kJ ± 1.1% | 53.3 kJ | Whole server 110 kJ |
| Parallel | dell | 3 × 9 | 12.2 s ± 0.2% | 680 W | 521 W | 724 W | 8.46 kJ ± 1.4% | 6.48 kJ | Whole server 8.46 kJ |
| GPU | ampere | 3 × 1st of 17 | 5 s ± 0.6% | 546 W | 234 W | 602 W | 2.78 kJ ± 4.6% | 1.19 kJ | CPU socket 0 395 J, CPU socket 1 305 J, GPU 0, in use 1.2 kJ, GPUs 1–3, unused 887 J |
| Hybrid | ampere | 3 × 1st of 18 | 5.08 s ± 0.1% | 549 W | 237 W | 656 W | 2.96 kJ ± 1.5% | 1.28 kJ | CPU socket 0 493 J, CPU socket 1 323 J, GPU 0, in use 1.23 kJ, GPUs 1–3, unused 919 J |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 3 × 1 | 2,610 s ± 0.1% | 258 W | 148 W | 275 W | 672 kJ ± 0.1% | 385 kJ | Whole server 672 kJ |
| Serial-Opt | epyc | 3 × 1 | 279 s ± 0.5% | 261 W | 151 W | 267 W | 72.9 kJ ± 0.9% | 42.2 kJ | Whole server 72.9 kJ |
| Parallel | epyc | 3 × 15 | 7.77 s ± 0.2% | 597 W | 487 W | 619 W | 4.71 kJ ± 0.4% | 3.84 kJ | Whole server 4.71 kJ |
| Serial-Simple | dell | 3 × 1 | 2,815 s ± 0.1% | 307 W | 149 W | 338 W | 866 kJ ± 0.2% | 419 kJ | Whole server 866 kJ |
| Serial-Opt | dell | 3 × 1 | 436 s ± 1.0% | 309 W | 150 W | 313 W | 135 kJ ± 0.8% | 65.6 kJ | Whole server 135 kJ |
| Parallel | dell | 3 × 7 | 15.4 s ± 0.1% | 666 W | 508 W | 745 W | 10.4 kJ ± 1.4% | 7.93 kJ | Whole server 10.4 kJ |
| GPU | ampere | 3 × 1st of 17 | 5.04 s ± 0.5% | 536 W | 224 W | 601 W | 2.75 kJ ± 7.5% | 1.15 kJ | CPU socket 0 397 J, CPU socket 1 307 J, GPU 0, in use 1.17 kJ, GPUs 1–3, unused 872 J |
| Hybrid | ampere | 3 × 1st of 18 | 5.1 s ± 0.4% | 559 W | 247 W | 645 W | 3.03 kJ ± 1.7% | 1.34 kJ | CPU socket 0 494 J, CPU socket 1 325 J, GPU 0, in use 1.26 kJ, GPUs 1–3, unused 956 J |

### Later trees in the same process (ampere)

| Implementation | Algorithm | first tree: t, E | later trees: t, E (mean) |
|---|---|---|---|
| GPU | CART | 5 s, 2.78 kJ | 7.11 s, 3.21 kJ |
| GPU | C4.5 | 5.04 s, 2.75 kJ | 7.14 s, 3.22 kJ |
| Hybrid | CART | 5.08 s, 2.96 kJ | 6.93 s, 3.31 kJ |
| Hybrid | C4.5 | 5.1 s, 3.03 kJ | 6.95 s, 3.33 kJ |
