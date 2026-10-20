# Profile numbers

Seconds of one profiled run per case (`train total`), from `profiles/*.json`.

## Phases of the training thread (s)

| Case | Run | setup | upload | presort | gpu | cpu | arrange | prune | other | train total | without --profile |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | gpuonly-hopper-ecbdl_15m | 0.04 | 0.70 | 0.53 | 6.73 | 0.00 | 0.00 | 0.00 | 0.02 | 8.01 | 7.99 |
| cart_alpha_nodepth | gpuonly-ampere-ecbdl_15m | 0.04 | 1.80 | 0.68 | 9.12 | 0.00 | 0.00 | 0.00 | 0.02 | 11.67 | 11.64 |
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.04 | 0.70 | 0.53 | 6.75 | 0.00 | 0.00 | 0.00 | 0.02 | 8.03 | 8.00 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.04 | 1.72 | 0.68 | 9.12 | 0.00 | 0.00 | 0.00 | 0.03 | 11.59 | 11.69 |
| cart_full_nodepth | gpuonly-hopper-ecbdl_15m | 0.04 | 0.70 | 0.53 | 25.89 | 0.00 | 0.00 | 0.00 | 0.03 | 27.18 | 27.02 |
| cart_full_nodepth | gpuonly-ampere-ecbdl_15m | 0.08 | 1.81 | 0.69 | 32.26 | 0.00 | 0.00 | 0.00 | 0.04 | 34.88 | 34.86 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.05 | 0.70 | 0.53 | 15.91 | 0.02 | 0.00 | 0.00 | 0.03 | 17.24 | 17.18 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.05 | 1.76 | 0.68 | 21.20 | 0.00 | 0.00 | 0.00 | 0.05 | 23.75 | 23.72 |
| c45 | gpuonly-hopper-ecbdl_15m | 0.04 | 0.70 | 0.61 | 35.89 | 0.00 | 2.05 | 20.83 | 0.06 | 60.17 | 60.06 |
| c45 | gpuonly-ampere-ecbdl_15m | 0.08 | 1.80 | 0.78 | 46.69 | 0.00 | 2.42 | 31.19 | 0.08 | 83.03 | 83.01 |
| c45 | cuda-hopper-ecbdl_15m | 0.04 | 0.70 | 0.61 | 25.70 | 0.12 | 1.78 | 17.47 | 0.12 | 46.53 | 46.34 |
| c45 | cuda-ampere-ecbdl_15m | 0.08 | 1.71 | 0.78 | 33.99 | 0.13 | 2.49 | 26.01 | 0.09 | 65.28 | 65.61 |

## CPU threads (thread-seconds)

| Case | Run | threads | sweep | partition | fused | presort | grow | feed | prune | other | idle |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | gpuonly-hopper-ecbdl_15m | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15.4 | 0.0 | 0.0 | 112.8 |
| cart_alpha_nodepth | gpuonly-ampere-ecbdl_15m | 24 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 31.8 | 0.0 | 0.0 | 248.3 |
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 32 | 1.9 | 0.8 | 0.0 | 0.0 | 0.0 | 16.3 | 0.0 | 0.0 | 238.0 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 48 | 1.4 | 0.8 | 0.0 | 0.0 | 0.0 | 31.1 | 0.0 | 0.0 | 523.0 |
| cart_full_nodepth | gpuonly-hopper-ecbdl_15m | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 34.7 | 0.0 | 0.0 | 400.2 |
| cart_full_nodepth | gpuonly-ampere-ecbdl_15m | 24 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 54.6 | 0.0 | 0.0 | 782.5 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 32 | 78.5 | 59.6 | 0.0 | 0.0 | 2.3 | 58.3 | 0.0 | 0.0 | 353.1 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 48 | 123.8 | 129.8 | 0.0 | 0.0 | 5.3 | 88.0 | 0.0 | 0.1 | 793.1 |
| c45 | gpuonly-hopper-ecbdl_15m | 16 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 44.7 | 360.5 | 0.1 | 557.4 |
| c45 | gpuonly-ampere-ecbdl_15m | 24 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 69.3 | 787.6 | 0.1 | 1135.6 |
| c45 | cuda-hopper-ecbdl_15m | 32 | 113.2 | 63.0 | 0.0 | 0.0 | 4.2 | 67.8 | 587.2 | 0.1 | 653.6 |
| c45 | cuda-ampere-ecbdl_15m | 48 | 128.1 | 108.6 | 0.0 | 0.0 | 5.2 | 104.7 | 1305.2 | 0.1 | 1481.8 |

## GPU (s busy)

| Case | Run | kernel presort | kernel search | kernel partition | kernel handoff | kernel leaves | copy upload | copy handoff | copy other | GPU levels | hand-offs (nodes, rows, GB) | host→GPU GB | GPU→host GB |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | gpuonly-hopper-ecbdl_15m | 0.52 | 2.04 | 4.46 | 0.00 | 0.00 | 0.70 | 0.00 | 0.10 | 53 | 0, 0, 0.0 | 37.9 | 0.0 |
| cart_alpha_nodepth | gpuonly-ampere-ecbdl_15m | 0.68 | 2.61 | 6.25 | 0.00 | 0.00 | 1.80 | 0.00 | 0.10 | 53 | 0, 0, 0.0 | 37.9 | 0.0 |
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 2.03 | 4.45 | 0.00 | 0.00 | 0.69 | 0.04 | 0.08 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 2.59 | 6.24 | 0.00 | 0.00 | 1.71 | 0.10 | 0.09 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_full_nodepth | gpuonly-hopper-ecbdl_15m | 0.52 | 8.83 | 11.20 | 0.00 | 0.00 | 0.70 | 0.00 | 1.68 | 94 | 0, 0, 0.0 | 37.9 | 0.0 |
| cart_full_nodepth | gpuonly-ampere-ecbdl_15m | 0.68 | 10.87 | 15.35 | 0.00 | 0.00 | 1.80 | 0.00 | 1.54 | 94 | 0, 0, 0.0 | 37.9 | 0.0 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 4.92 | 9.07 | 0.05 | 0.00 | 0.69 | 0.67 | 0.34 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 6.13 | 12.73 | 0.06 | 0.00 | 1.75 | 1.62 | 0.47 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| c45 | gpuonly-hopper-ecbdl_15m | 0.60 | 11.35 | 18.54 | 0.00 | 0.01 | 0.70 | 0.00 | 1.43 | 274 | 0, 0, 0.0 | 37.9 | 0.1 |
| c45 | gpuonly-ampere-ecbdl_15m | 0.77 | 14.05 | 25.42 | 0.00 | 0.01 | 1.80 | 0.00 | 1.46 | 274 | 0, 0, 0.0 | 37.9 | 0.1 |
| c45 | cuda-hopper-ecbdl_15m | 0.60 | 7.58 | 16.43 | 0.06 | 0.01 | 0.69 | 0.93 | 0.24 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
| c45 | cuda-ampere-ecbdl_15m | 0.77 | 9.49 | 22.82 | 0.07 | 0.01 | 1.71 | 2.42 | 0.25 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
