# Profile numbers

Seconds of one profiled run per case (`train total`), from `profiles/*.json`.

## Phases of the training thread (s)

| Case | Run | setup | upload | presort | gpu | cpu | arrange | prune | other | train total | without --profile |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | gpuonly-hopper-ecbdl_15m | 0.07 | 2.93 | 0.53 | 6.77 | 0.00 | 0.00 | 0.00 | 0.02 | 10.31 | 10.29 |
| cart_alpha_nodepth | gpuonly-ampere-ecbdl_15m | 0.07 | 2.22 | 0.68 | 9.11 | 0.00 | 0.00 | 0.00 | 0.02 | 12.11 | 12.06 |
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.05 | 0.70 | 0.53 | 6.74 | 0.00 | 0.00 | 0.00 | 0.02 | 8.04 | 7.99 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.09 | 1.78 | 0.68 | 9.11 | 0.00 | 0.00 | 0.00 | 0.03 | 11.69 | 11.66 |
| cart_full_nodepth | gpuonly-hopper-ecbdl_15m | 0.06 | 2.88 | 0.53 | 38.98 | 0.00 | 0.00 | 0.00 | 0.04 | 42.48 | 41.91 |
| cart_full_nodepth | gpuonly-ampere-ecbdl_15m | 0.07 | 2.22 | 0.68 | 31.96 | 0.00 | 0.00 | 0.00 | 0.05 | 34.99 | 34.91 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.07 | 0.70 | 0.53 | 16.01 | 0.02 | 0.00 | 0.00 | 0.03 | 17.35 | 17.40 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.10 | 1.79 | 0.68 | 21.49 | 0.00 | 0.00 | 0.00 | 0.04 | 24.12 | 23.91 |
| c45 | gpuonly-hopper-ecbdl_15m | 0.06 | 2.87 | 0.60 | 39.19 | 0.00 | 52.78 | 283.82 | 0.14 | 379.48 | 377.60 |
| c45 | gpuonly-ampere-ecbdl_15m | 0.07 | 2.22 | 0.78 | 46.35 | 0.00 | 18.23 | 160.14 | 0.10 | 227.90 | 225.95 |
| c45 | cuda-hopper-ecbdl_15m | 0.07 | 0.70 | 0.61 | 25.58 | 0.12 | 1.65 | 22.62 | 0.09 | 51.43 | 51.11 |
| c45 | cuda-ampere-ecbdl_15m | 0.18 | 1.70 | 0.78 | 34.10 | 0.14 | 2.50 | 24.74 | 0.11 | 64.26 | 65.41 |

## CPU threads (thread-seconds)

| Case | Run | threads | sweep | partition | fused | presort | grow | feed | prune | other | idle |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | gpuonly-hopper-ecbdl_15m | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.3 | 0.0 | 0.1 | 0.0 |
| cart_alpha_nodepth | gpuonly-ampere-ecbdl_15m | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 12.1 | 0.0 | 0.1 | 0.0 |
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 32 | 1.9 | 0.9 | 0.0 | 0.0 | 0.0 | 21.0 | 0.0 | 0.1 | 233.3 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 48 | 1.4 | 0.8 | 0.0 | 0.0 | 0.0 | 29.4 | 0.0 | 0.4 | 529.4 |
| cart_full_nodepth | gpuonly-hopper-ecbdl_15m | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 42.4 | 0.0 | 0.1 | 0.0 |
| cart_full_nodepth | gpuonly-ampere-ecbdl_15m | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 34.9 | 0.0 | 0.1 | 0.0 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 32 | 79.4 | 60.3 | 0.0 | 0.0 | 2.4 | 67.9 | 0.0 | 0.1 | 345.3 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 48 | 124.1 | 130.3 | 0.0 | 0.0 | 5.1 | 86.9 | 0.0 | 0.6 | 810.7 |
| c45 | gpuonly-hopper-ecbdl_15m | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 42.7 | 336.6 | 0.1 | 0.0 |
| c45 | gpuonly-ampere-ecbdl_15m | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 49.4 | 178.4 | 0.1 | 0.0 |
| c45 | cuda-hopper-ecbdl_15m | 32 | 114.2 | 63.2 | 0.0 | 0.0 | 4.1 | 77.9 | 745.3 | 0.1 | 641.3 |
| c45 | cuda-ampere-ecbdl_15m | 48 | 126.5 | 107.9 | 0.0 | 0.0 | 5.2 | 105.7 | 1246.0 | 0.6 | 1493.1 |

## GPU (s busy)

| Case | Run | kernel presort | kernel search | kernel partition | kernel handoff | kernel leaves | copy upload | copy handoff | copy other | GPU levels | hand-offs (nodes, rows, GB) | host→GPU GB | GPU→host GB |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | gpuonly-hopper-ecbdl_15m | 0.52 | 2.04 | 4.46 | 0.00 | 0.00 | 0.69 | 0.00 | 0.12 | 53 | 0, 0, 0.0 | 37.9 | 0.0 |
| cart_alpha_nodepth | gpuonly-ampere-ecbdl_15m | 0.68 | 2.61 | 6.25 | 0.00 | 0.00 | 1.56 | 0.00 | 0.09 | 53 | 0, 0, 0.0 | 37.9 | 0.0 |
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 2.04 | 4.47 | 0.00 | 0.00 | 0.69 | 0.04 | 0.07 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 2.59 | 6.24 | 0.00 | 0.00 | 1.77 | 0.10 | 0.07 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_full_nodepth | gpuonly-hopper-ecbdl_15m | 0.52 | 8.83 | 11.19 | 0.00 | 0.00 | 0.69 | 0.00 | 2.36 | 94 | 0, 0, 0.0 | 37.9 | 0.0 |
| cart_full_nodepth | gpuonly-ampere-ecbdl_15m | 0.68 | 10.87 | 15.34 | 0.00 | 0.00 | 1.56 | 0.00 | 1.48 | 94 | 0, 0, 0.0 | 37.9 | 0.0 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 4.91 | 9.05 | 0.05 | 0.00 | 0.69 | 0.69 | 0.38 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 6.14 | 12.73 | 0.06 | 0.00 | 1.78 | 1.59 | 0.46 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| c45 | gpuonly-hopper-ecbdl_15m | 0.60 | 11.41 | 18.64 | 0.00 | 0.01 | 0.69 | 0.00 | 1.99 | 274 | 0, 0, 0.0 | 37.9 | 0.1 |
| c45 | gpuonly-ampere-ecbdl_15m | 0.77 | 14.03 | 25.43 | 0.00 | 0.01 | 1.56 | 0.00 | 1.41 | 274 | 0, 0, 0.0 | 37.9 | 0.1 |
| c45 | cuda-hopper-ecbdl_15m | 0.60 | 7.52 | 16.32 | 0.06 | 0.01 | 0.69 | 0.93 | 0.27 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
| c45 | cuda-ampere-ecbdl_15m | 0.77 | 9.50 | 22.83 | 0.07 | 0.01 | 1.70 | 2.39 | 0.22 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
