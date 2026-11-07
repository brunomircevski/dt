### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,332 s ± 0.0% | 256 W | 140 W | 258 W | 598 kJ ± 0.0% | 327 kJ | CPU socket 0 216 kJ, CPU socket 1 174 kJ, Rest of the server 208 kJ |
| Serial-Opt | epyc | 3 × 1 | 225 s ± 0.5% | 264 W | 147 W | 277 W | 59.4 kJ ± 1.3% | 33.2 kJ | CPU socket 0 19.8 kJ, CPU socket 1 19 kJ, Rest of the server 20.6 kJ |
| Parallel | epyc | 10 × 20 | 6.44 s ± 0.4% | 595 W | 479 W | 622 W | 3.83 kJ ± 1.3% | 3.08 kJ | CPU socket 0 1.4 kJ, CPU socket 1 1.33 kJ, Rest of the server 1.1 kJ |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | epyc | 1 × 1 | 2,639 s ± 0.0% | 257 W | 141 W | 264 W | 679 kJ ± 0.0% | 372 kJ | CPU socket 0 218 kJ, CPU socket 1 223 kJ, Rest of the server 238 kJ |
| Serial-Opt | epyc | 3 × 1 | 278 s ± 0.2% | 263 W | 147 W | 272 W | 73 kJ ± 1.1% | 40.8 kJ | CPU socket 0 24.3 kJ, CPU socket 1 23.2 kJ, Rest of the server 25.6 kJ |
| Parallel | epyc | 10 × 16 | 7.91 s ± 0.3% | 601 W | 485 W | 627 W | 4.75 kJ ± 1.4% | 3.83 kJ | CPU socket 0 1.72 kJ, CPU socket 1 1.67 kJ, Rest of the server 1.36 kJ |

### Time check

Time per tree (./tree's train time) against: the first tree of each process, the training windows the energy is summed over, and the time benchmark (cluster-ecbdl, ecbdl_15m). Flagged (!) above 3%.

| Implementation | Node | Algorithm | t per tree | first tree | window | time benchmark |
|---|---|---|---|---|---|---|
| Serial-Simple | epyc | CART | 2,332 s | 2,332 s (+0.0%) | 2,332 s (-0.0%) | 2,273 s (+2.6%) |
| Serial-Simple | epyc | C4.5 | 2,639 s | 2,639 s (+0.0%) | 2,639 s (-0.0%) | 2,587 s (+2.0%) |
| Serial-Opt | epyc | CART | 225 s | 225 s (+0.0%) | 225 s (-0.0%) | 225 s (+0.2%) |
| Serial-Opt | epyc | C4.5 | 278 s | 278 s (+0.0%) | 278 s (-0.0%) | 277 s (+0.2%) |
| Parallel | epyc | CART | 6.44 s | 6.44 s (-0.0%) | 6.44 s (-0.0%) | 6.32 s (+1.9%) |
| Parallel | epyc | C4.5 | 7.91 s | 7.88 s (+0.3%) | 7.91 s (-0.0%) | 7.81 s (+1.3%) |
