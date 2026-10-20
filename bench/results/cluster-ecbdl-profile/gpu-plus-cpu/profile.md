# Profile numbers

Seconds of one profiled run per case (`train total`), from `profiles/*.json`.

## Phases of the training thread (s)

| Case | Run | setup | upload | presort | gpu | cpu | arrange | prune | other | train total | without --profile |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.05 | 0.70 | 0.53 | 6.74 | 0.00 | 0.00 | 0.00 | 0.02 | 8.04 | 7.99 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.09 | 1.78 | 0.68 | 9.11 | 0.00 | 0.00 | 0.00 | 0.03 | 11.69 | 11.66 |
| cart_alpha_nodepth | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.63 | 0.00 | 14.90 | 0.00 | 0.00 | 0.00 | 16.53 | 16.81 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.07 | 0.70 | 0.53 | 16.01 | 0.02 | 0.00 | 0.00 | 0.03 | 17.35 | 17.40 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.10 | 1.79 | 0.68 | 21.49 | 0.00 | 0.00 | 0.00 | 0.04 | 24.12 | 23.91 |
| cart_full_nodepth | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.40 | 0.00 | 33.11 | 0.00 | 0.00 | 0.01 | 34.51 | 35.35 |
| c45 | cuda-hopper-ecbdl_15m | 0.07 | 0.70 | 0.61 | 25.58 | 0.12 | 1.65 | 22.62 | 0.09 | 51.43 | 51.11 |
| c45 | cuda-ampere-ecbdl_15m | 0.18 | 1.70 | 0.78 | 34.10 | 0.14 | 2.50 | 24.74 | 0.11 | 64.26 | 65.41 |
| c45 | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.90 | 0.00 | 65.23 | 0.48 | 10.36 | 0.01 | 77.99 | 79.29 |

## CPU threads (thread-seconds)

| Case | Run | threads | sweep | partition | fused | presort | grow | feed | prune | other | idle |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 32 | 1.9 | 0.9 | 0.0 | 0.0 | 0.0 | 21.0 | 0.0 | 0.1 | 233.3 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 48 | 1.4 | 0.8 | 0.0 | 0.0 | 0.0 | 29.4 | 0.0 | 0.4 | 529.4 |
| cart_alpha_nodepth | cpu-epyc-ecbdl_15m | 48 | 63.9 | 68.1 | 543.9 | 76.9 | 8.9 | 0.0 | 0.0 | 0.5 | 31.2 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 32 | 79.4 | 60.3 | 0.0 | 0.0 | 2.4 | 67.9 | 0.0 | 0.1 | 345.3 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 48 | 124.1 | 130.3 | 0.0 | 0.0 | 5.1 | 86.9 | 0.0 | 0.6 | 810.7 |
| cart_full_nodepth | cpu-epyc-ecbdl_15m | 48 | 299.9 | 402.7 | 819.8 | 66.2 | 15.7 | 0.0 | 0.0 | 0.5 | 52.0 |
| c45 | cuda-hopper-ecbdl_15m | 32 | 114.2 | 63.2 | 0.0 | 0.0 | 4.1 | 77.9 | 745.3 | 0.1 | 641.3 |
| c45 | cuda-ampere-ecbdl_15m | 48 | 126.5 | 107.9 | 0.0 | 0.0 | 5.2 | 105.7 | 1246.0 | 0.6 | 1493.1 |
| c45 | cpu-epyc-ecbdl_15m | 48 | 284.0 | 285.9 | 2447.5 | 86.5 | 39.3 | 0.0 | 489.0 | 0.9 | 110.5 |

## GPU (s busy)

| Case | Run | kernel presort | kernel search | kernel partition | kernel handoff | kernel leaves | copy upload | copy handoff | copy other | GPU levels | hand-offs (nodes, rows, GB) | host→GPU GB | GPU→host GB |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 2.04 | 4.47 | 0.00 | 0.00 | 0.69 | 0.04 | 0.07 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 2.59 | 6.24 | 0.00 | 0.00 | 1.77 | 0.10 | 0.07 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 4.91 | 9.05 | 0.05 | 0.00 | 0.69 | 0.69 | 0.38 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 6.14 | 12.73 | 0.06 | 0.00 | 1.78 | 1.59 | 0.46 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| c45 | cuda-hopper-ecbdl_15m | 0.60 | 7.52 | 16.32 | 0.06 | 0.01 | 0.69 | 0.93 | 0.27 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
| c45 | cuda-ampere-ecbdl_15m | 0.77 | 9.50 | 22.83 | 0.07 | 0.01 | 1.70 | 2.39 | 0.22 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
