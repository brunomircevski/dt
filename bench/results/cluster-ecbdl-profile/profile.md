# Profile numbers

Seconds of one profiled run per case (`train total`), from `profiles/*.json`.

## Phases of the training thread (s)

| Case | Run | setup | upload | presort | gpu | cpu | arrange | prune | other | train total | without --profile |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.05 | 0.70 | 0.53 | 6.73 | 0.00 | 0.00 | 0.00 | 0.02 | 8.03 | 8.00 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.10 | 1.73 | 0.68 | 9.14 | 0.00 | 0.00 | 0.00 | 0.02 | 11.67 | 11.64 |
| cart_alpha_nodepth | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.66 | 0.00 | 15.14 | 0.00 | 0.00 | 0.00 | 16.80 | 17.06 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.05 | 0.70 | 0.53 | 16.97 | 0.02 | 0.00 | 0.00 | 0.03 | 18.29 | 18.21 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.11 | 1.75 | 0.68 | 23.00 | 0.00 | 0.00 | 0.00 | 0.04 | 25.59 | 25.72 |
| cart_full_nodepth | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.59 | 0.00 | 33.83 | 0.00 | 0.00 | 0.01 | 35.43 | 35.55 |
| c45 | cuda-hopper-ecbdl_15m | 0.04 | 0.70 | 0.61 | 26.56 | 0.14 | 1.27 | 17.53 | 0.09 | 46.94 | 46.88 |
| c45 | cuda-ampere-ecbdl_15m | 1.97 | 1.68 | 0.78 | 33.90 | 0.13 | 1.80 | 26.20 | 0.10 | 66.56 | 66.85 |
| c45 | cpu-epyc-ecbdl_15m | 0.00 | 0.00 | 1.90 | 0.00 | 65.61 | 0.78 | 10.61 | 0.06 | 78.95 | 77.37 |

## CPU threads (thread-seconds)

| Case | Run | threads | sweep | partition | fused | presort | grow | feed | prune | other | idle |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 32 | 2.6 | 1.1 | 0.0 | 0.0 | 0.0 | 17.2 | 0.0 | 0.2 | 235.9 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 48 | 1.3 | 0.7 | 0.0 | 0.0 | 0.0 | 31.8 | 0.0 | 0.4 | 526.1 |
| cart_alpha_nodepth | cpu-epyc-ecbdl_15m | 48 | 65.0 | 68.4 | 542.6 | 76.3 | 8.7 | 0.0 | 0.0 | 0.8 | 44.6 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 32 | 83.8 | 60.6 | 0.0 | 0.0 | 2.4 | 44.2 | 0.0 | 0.1 | 394.3 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 48 | 125.5 | 130.8 | 0.0 | 0.0 | 5.2 | 73.2 | 0.0 | 0.5 | 893.2 |
| cart_full_nodepth | cpu-epyc-ecbdl_15m | 48 | 311.8 | 424.7 | 819.4 | 73.6 | 15.1 | 0.0 | 0.0 | 0.7 | 55.4 |
| c45 | cuda-hopper-ecbdl_15m | 32 | 114.9 | 64.9 | 0.0 | 0.0 | 4.2 | 52.4 | 580.0 | 0.2 | 685.5 |
| c45 | cuda-ampere-ecbdl_15m | 48 | 123.1 | 108.5 | 0.0 | 0.0 | 5.2 | 84.9 | 1280.3 | 2.5 | 1591.0 |
| c45 | cpu-epyc-ecbdl_15m | 48 | 278.6 | 291.6 | 2453.3 | 86.4 | 37.1 | 0.0 | 496.9 | 1.0 | 145.0 |

## GPU (s busy)

| Case | Run | kernel presort | kernel search | kernel partition | kernel handoff | kernel leaves | copy upload | copy handoff | copy other | GPU levels | hand-offs (nodes, rows, GB) | host→GPU GB | GPU→host GB |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cart_alpha_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 2.03 | 4.45 | 0.00 | 0.00 | 0.69 | 0.04 | 0.07 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_alpha_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 2.59 | 6.24 | 0.00 | 0.00 | 1.72 | 0.13 | 0.09 | 53 | 236, 307,355, 1.6 | 37.9 | 1.6 |
| cart_full_nodepth | cuda-hopper-ecbdl_15m | 0.53 | 4.91 | 9.06 | 0.05 | 0.00 | 0.69 | 0.54 | 1.53 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| cart_full_nodepth | cuda-ampere-ecbdl_15m | 0.68 | 6.14 | 12.73 | 0.06 | 0.00 | 1.74 | 2.02 | 0.39 | 91 | 49,187, 5,905,501, 29.8 | 37.9 | 29.8 |
| c45 | cuda-hopper-ecbdl_15m | 0.60 | 7.52 | 16.33 | 0.06 | 0.01 | 0.69 | 0.94 | 0.20 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
| c45 | cuda-ampere-ecbdl_15m | 0.78 | 9.50 | 22.82 | 0.07 | 0.01 | 1.67 | 2.97 | 0.18 | 272 | 21,972, 8,838,665, 44.6 | 37.9 | 44.7 |
