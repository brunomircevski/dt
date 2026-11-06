### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,312 s ± 0.0% | 257 W | 134 W | 259 W | 594 kJ ± 0.0% | 309 kJ | CPU socket 0 190 kJ, CPU socket 1 195 kJ, Rest of the server 209 kJ |
| Serial-Opt | epyc | 3 × 1 | 227 s ± 0.2% | 265 W | 141 W | 278 W | 60.2 kJ ± 0.7% | 32.1 kJ | CPU socket 0 22 kJ, CPU socket 1 17.1 kJ, Rest of the server 21.1 kJ |
| Parallel | epyc | 10 × 20 | 6.29 s ± 0.1% | 606 W | 483 W | 623 W | 3.88 kJ ± 1.6% | 3.09 kJ | CPU socket 0 1.39 kJ, CPU socket 1 1.34 kJ, Rest of the server 1.16 kJ |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,609 s ± 0.0% | 259 W | 135 W | 264 W | 675 kJ ± 0.0% | 352 kJ | CPU socket 0 216 kJ, CPU socket 1 221 kJ, Rest of the server 238 kJ |
| Serial-Opt | epyc | 3 × 1 | 278 s ± 0.7% | 265 W | 142 W | 272 W | 73.9 kJ ± 1.5% | 39.5 kJ | CPU socket 0 24.4 kJ, CPU socket 1 23.3 kJ, Rest of the server 26.2 kJ |
| Parallel | epyc | 10 × 16 | 7.79 s ± 0.1% | 612 W | 488 W | 628 W | 4.83 kJ ± 1.5% | 3.86 kJ | CPU socket 0 1.73 kJ, CPU socket 1 1.67 kJ, Rest of the server 1.44 kJ |
