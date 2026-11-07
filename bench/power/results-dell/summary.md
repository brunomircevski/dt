### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | dell | 1 × 1 | 2,576 s ± 0.0% | 307 W | 152 W | 311 W | 792 kJ ± 0.0% | 392 kJ | CPU socket 0 278 kJ, CPU socket 1 231 kJ, Rest of the server 282 kJ |
| Serial-Opt | dell | 3 × 1 | 359 s ± 1.2% | 309 W | 153 W | 314 W | 111 kJ ± 1.3% | 55.1 kJ | CPU socket 0 38.6 kJ, CPU socket 1 32.8 kJ, Rest of the server 39.4 kJ |
| Parallel | dell | 10 × 11 | 12.6 s ± 0.7% | 676 W | 521 W | 738 W | 8.54 kJ ± 1.5% | 6.58 kJ | CPU socket 0 3.03 kJ, CPU socket 1 3.02 kJ, Rest of the server 2.49 kJ |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | dell | 1 × 1 | 2,865 s ± 0.0% | 308 W | 153 W | 316 W | 883 kJ ± 0.0% | 438 kJ | CPU socket 0 264 kJ, CPU socket 1 300 kJ, Rest of the server 318 kJ |
| Serial-Opt | dell | 3 × 1 | 441 s ± 0.2% | 309 W | 153 W | 314 W | 136 kJ ± 0.3% | 67.6 kJ | CPU socket 0 45.4 kJ, CPU socket 1 42.3 kJ, Rest of the server 48.4 kJ |
| Parallel | dell | 10 × 9 | 15.9 s ± 1.1% | 663 W | 508 W | 754 W | 10.6 kJ ± 1.9% | 8.1 kJ | CPU socket 0 3.75 kJ, CPU socket 1 3.73 kJ, Rest of the server 3.1 kJ |

### Time check

Time per tree (./tree's train time) against: the first tree of each process, the training windows the energy is summed over, and the time benchmark (cluster-ecbdl, ecbdl_15m). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | window | time benchmark |
|---|---|---|---|---|---|---|
| Serial-Simple | dell | CART | 2,576 s | 2,576 s (+0.0%) | 2,576 s (-0.0%) | – |
| Serial-Simple | dell | C4.5 | 2,865 s | 2,865 s (+0.0%) | 2,865 s (-0.0%) | – |
| Serial-Opt | dell | CART | 359 s | 359 s (+0.0%) | 359 s (-0.0%) | – |
| Serial-Opt | dell | C4.5 | 441 s | 441 s (+0.0%) | 441 s (-0.0%) | – |
| Parallel | dell | CART | 12.6 s | 12.9 s (-2.1%) | 12.6 s (-0.0%) | – |
| Parallel | dell | C4.5 | 15.9 s | 16.2 s (-1.4%) | 15.9 s (-0.0%) | – |
