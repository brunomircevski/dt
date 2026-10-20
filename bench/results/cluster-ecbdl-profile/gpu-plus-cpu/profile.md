# Profile numbers

Seconds of one profiled run per case (`train total`), from `profiles/*.json`.

## Phases of the training thread (s)

| Case | Run | setup | upload | presort | gpu | cpu | arrange | prune | other | train total | without --profile |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.04 | 0.70 | 0.53 | 6.75 | 0.00 | 0.00 | 0.00 | 0.02 | 8.03 | 8.00 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.04 | 1.72 | 0.68 | 9.12 | 0.00 | 0.00 | 0.00 | 0.03 | 11.59 | 11.69 |
| cart_alpha_nodepth | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.39 | 0.00 | 14.54 | 0.00 | 0.00 | 0.00 | 15.93 | 15.92 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.05 | 0.70 | 0.53 | 15.91 | 0.02 | 0.00 | 0.00 | 0.03 | 17.24 | 17.18 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.05 | 1.76 | 0.68 | 21.20 | 0.00 | 0.00 | 0.00 | 0.05 | 23.75 | 23.72 |
| cart_full_nodepth | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.39 | 0.00 | 32.87 | 0.00 | 0.00 | 0.00 | 34.26 | 34.26 |
| c45 | cuda-hopper-ecbdl_15m | 0.04 | 0.70 | 0.61 | 25.70 | 0.12 | 1.78 | 17.47 | 0.12 | 46.53 | 46.34 |
| c45 | cuda-ampere-ecbdl_15m | 0.08 | 1.71 | 0.78 | 33.99 | 0.13 | 2.49 | 26.01 | 0.09 | 65.28 | 65.61 |
| c45 | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.56 | 0.00 | 64.40 | 0.50 | 9.59 | 0.01 | 76.07 | 76.08 |

## CPU threads (thread-seconds)

| Case | Run | threads | sweep | partition | fused | presort | grow | feed | prune | other | idle |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 32 | 1.9 | 0.8 | 0.0 | 0.0 | 0.0 | 16.3 | 0.0 | 0.0 | 238.0 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 48 | 1.4 | 0.8 | 0.0 | 0.0 | 0.0 | 31.1 | 0.0 | 0.0 | 523.0 |
| cart_alpha_nodepth | cpu-epyc-ecbdl_15m | 48 | 60.8 | 65.4 | 540.3 | 65.9 | 8.9 | 0.0 | 0.0 | 0.4 | 23.1 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 32 | 78.5 | 59.6 | 0.0 | 0.0 | 2.3 | 58.3 | 0.0 | 0.0 | 353.1 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 48 | 123.8 | 129.8 | 0.0 | 0.0 | 5.3 | 88.0 | 0.0 | 0.1 | 793.1 |
| cart_full_nodepth | cpu-epyc-ecbdl_15m | 48 | 299.4 | 403.6 | 817.8 | 65.8 | 14.9 | 0.0 | 0.0 | 0.3 | 42.9 |
| c45 | cuda-hopper-ecbdl_15m | 32 | 113.2 | 63.0 | 0.0 | 0.0 | 4.2 | 67.8 | 587.2 | 0.1 | 653.6 |
| c45 | cuda-ampere-ecbdl_15m | 48 | 128.1 | 108.6 | 0.0 | 0.0 | 5.2 | 104.7 | 1305.2 | 0.1 | 1481.8 |
| c45 | cpu-epyc-ecbdl_15m | 48 | 279.5 | 280.3 | 2426.4 | 74.2 | 38.7 | 0.0 | 450.4 | 0.3 | 101.9 |

## GPU (s busy)

| Case | Run | kernel presort | kernel search | kernel partition | kernel handoff | kernel leaves | copy upload | copy handoff | copy other | GPU levels | hand-offs (nodes, rows, GB) | host→GPU GB | GPU→host GB |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 2.03 | 4.45 | 0.00 | 0.00 | 0.69 | 0.04 | 0.08 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 2.59 | 6.24 | 0.00 | 0.00 | 1.71 | 0.10 | 0.09 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 4.92 | 9.07 | 0.05 | 0.00 | 0.69 | 0.67 | 0.34 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 6.13 | 12.73 | 0.06 | 0.00 | 1.75 | 1.62 | 0.47 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| c45 | cuda-hopper-ecbdl_15m | 0.60 | 7.58 | 16.43 | 0.06 | 0.01 | 0.69 | 0.93 | 0.24 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
| c45 | cuda-ampere-ecbdl_15m | 0.77 | 9.49 | 22.82 | 0.07 | 0.01 | 1.71 | 2.42 | 0.25 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
