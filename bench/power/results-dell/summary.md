### CART (gini), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | dell | 1 × 1 | 2,591 s ± 0.0% | 308 W | 153 W | 313 W | 798 kJ ± 0.0% | 398 kJ | CPU socket 0 239 kJ, CPU socket 1 271 kJ, Rest of the server 288 kJ |
| Serial-Opt | dell | 3 × 1 | 358 s ± 0.4% | 309 W | 155 W | 313 W | 111 kJ ± 0.8% | 55.5 kJ | CPU socket 0 38.6 kJ, CPU socket 1 32.9 kJ, Rest of the server 39.4 kJ |
| Parallel | dell | 10 × 10 | 12.3 s ± 0.2% | 684 W | 529 W | 737 W | 8.55 kJ ± 0.9% | 6.62 kJ | CPU socket 0 3.01 kJ, CPU socket 1 2.99 kJ, Rest of the server 2.54 kJ |

### C4.5 (gain ratio), depth ≤ 10, no pruning

| Implementation | Node | runs × trees measured | t per tree | mean P | P above idle | peak P | E per tree | E above idle | split of E | 
|---|---|---|---|---|---|---|---|---|---|
| Serial-Simple | dell | 1 × 1 | 2,859 s ± 0.0% | 308 W | 153 W | 315 W | 879 kJ ± 0.0% | 438 kJ | CPU socket 0 264 kJ, CPU socket 1 299 kJ, Rest of the server 316 kJ |
| Serial-Opt | dell | 3 × 1 | 434 s ± 0.6% | 310 W | 156 W | 315 W | 135 kJ ± 0.3% | 67.6 kJ | CPU socket 0 40.3 kJ, CPU socket 1 45.6 kJ, Rest of the server 48.6 kJ |
| Parallel | dell | 10 × 8 | 15.4 s ± 0.6% | 666 W | 512 W | 756 W | 10.4 kJ ± 1.9% | 8 kJ | CPU socket 0 3.69 kJ, CPU socket 1 3.67 kJ, Rest of the server 3.05 kJ |
