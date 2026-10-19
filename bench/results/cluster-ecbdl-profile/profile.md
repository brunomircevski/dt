# Profile numbers

Seconds of one profiled run per case (`train total`), from `profiles/*.json`.

## Phases of the training thread (s)

| Case | Run | setup | upload | presort | gpu | cpu | arrange | prune | other | train total | without --profile |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.05 | 0.70 | 0.53 | 6.74 | 0.00 | 0.00 | 0.00 | 0.02 | 8.04 | 8.02 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.08 | 1.74 | 0.68 | 9.14 | 0.00 | 0.00 | 0.00 | 0.03 | 11.68 | 11.57 |
| cart_alpha_nodepth | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.63 | 0.00 | 14.90 | 0.00 | 0.00 | 0.00 | 16.53 | 16.81 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.04 | 0.70 | 0.53 | 17.16 | 0.01 | 0.00 | 0.00 | 0.03 | 18.48 | 18.39 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.15 | 1.69 | 0.69 | 23.10 | 0.00 | 0.00 | 0.00 | 0.04 | 25.67 | 25.56 |
| cart_full_nodepth | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.40 | 0.00 | 33.11 | 0.00 | 0.00 | 0.01 | 34.51 | 35.35 |
| c45 | cuda-hopper-ecbdl_15m | 1.25 | 0.70 | 0.61 | 25.34 | 0.10 | 1.10 | 17.70 | 0.09 | 46.88 | 46.74 |
| c45 | cuda-ampere-ecbdl_15m | 2.02 | 1.78 | 0.78 | 33.96 | 0.13 | 1.82 | 26.46 | 0.10 | 67.05 | 65.89 |
| c45 | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.90 | 0.00 | 65.23 | 0.48 | 10.36 | 0.01 | 77.99 | 79.29 |

## CPU threads (thread-seconds)

| Case | Run | threads | sweep | partition | fused | presort | grow | feed | prune | other | idle |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 32 | 2.5 | 1.1 | 0.0 | 0.0 | 0.0 | 17.0 | 0.0 | 0.1 | 236.6 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 48 | 1.3 | 0.7 | 0.0 | 0.0 | 0.0 | 31.8 | 0.0 | 0.4 | 526.4 |
| cart_alpha_nodepth | cpu-epyc-ecbdl_15m | 48 | 63.9 | 68.1 | 543.9 | 76.9 | 8.9 | 0.0 | 0.0 | 0.5 | 31.2 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 32 | 82.9 | 61.6 | 0.0 | 0.0 | 2.4 | 44.6 | 0.0 | 0.1 | 399.7 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 48 | 123.0 | 130.5 | 0.0 | 0.0 | 5.3 | 74.4 | 0.0 | 0.7 | 898.3 |
| cart_full_nodepth | cpu-epyc-ecbdl_15m | 48 | 299.9 | 402.7 | 819.8 | 66.2 | 15.7 | 0.0 | 0.0 | 0.5 | 52.0 |
| c45 | cuda-hopper-ecbdl_15m | 32 | 114.6 | 64.8 | 0.0 | 0.0 | 4.2 | 50.9 | 580.0 | 1.4 | 684.3 |
| c45 | cuda-ampere-ecbdl_15m | 48 | 122.8 | 106.1 | 0.0 | 0.0 | 5.1 | 84.3 | 1298.9 | 2.5 | 1599.3 |
| c45 | cpu-epyc-ecbdl_15m | 48 | 284.0 | 285.9 | 2447.5 | 86.5 | 39.3 | 0.0 | 489.0 | 0.9 | 110.5 |

## GPU (s busy)

| Case | Run | kernel presort | kernel search | kernel partition | kernel handoff | kernel leaves | copy upload | copy handoff | copy other | GPU levels | hand-offs (nodes, rows, GB) | host→GPU GB | GPU→host GB |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.52 | 2.04 | 4.47 | 0.00 | 0.00 | 0.69 | 0.04 | 0.07 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 2.59 | 6.24 | 0.00 | 0.00 | 1.73 | 0.13 | 0.09 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 4.91 | 9.05 | 0.05 | 0.00 | 0.69 | 0.55 | 1.64 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 6.13 | 12.73 | 0.07 | 0.00 | 1.69 | 1.97 | 0.59 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| c45 | cuda-hopper-ecbdl_15m | 0.60 | 7.58 | 16.43 | 0.06 | 0.01 | 0.69 | 0.89 | 0.17 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
| c45 | cuda-ampere-ecbdl_15m | 0.77 | 9.49 | 22.81 | 0.07 | 0.01 | 1.77 | 2.99 | 0.22 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
