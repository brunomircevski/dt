| backend | dataset | protocol | double s | float s | float vs double | same tree | nodes d/f | test acc d / f (pp) |
|---|---|---|---:|---:|---:|---|---|---|
| multicore | covertype | cart_depth10 | 0.030 | 0.028 | -8.6% | yes | 1499/1499 | 77.6622 / 77.6622 (+0.0000) |
| multicore | covertype | c45_depth10_unpruned | 0.030 | 0.029 | -2.7% | yes | 723/723 | 59.9327 / 59.9327 (+0.0000) |
| multicore | covertype | cart_full | 0.051 | 0.056 | +9.2% | no | 48095/48233 | 93.8788 / 93.8702 (-0.0086) |
| multicore | covertype | c45_unpruned | 0.078 | 0.070 | -10.8% | no | 30321/30335 | 94.4467 / 94.4450 (-0.0017) |
| multicore | susy | cart_depth10 | 0.180 | 0.219 | +21.9% | no | 1959/1965 | 79.0082 / 79.0106 (+0.0024) |
| multicore | susy | c45_depth10_unpruned | 0.167 | 0.175 | +4.7% | no | 743/769 | 78.6472 / 78.6490 (+0.0018) |
| multicore | susy | cart_full | 0.295 | 115.047 | +38951.9% | no | 1194859/1443835 | 71.5684 / 71.5842 (+0.0158) |
| multicore | susy | c45_unpruned | 0.312 | 0.331 | +6.3% | no | 24683/24593 | 79.4952 / 79.5024 (+0.0072) |
| multicore | higgs | cart_depth10 | 0.359 | 0.451 | +25.6% | no | 2029/2029 | 70.5172 / 70.5136 (-0.0036) |
| multicore | higgs | c45_depth10_unpruned | 0.390 | 0.382 | -2.2% | no | 935/927 | 63.7296 / 63.7340 (+0.0044) |
| multicore | higgs | cart_full | 0.850 | 1.050 | +23.5% | no | 2864695/2864913 | 67.3570 / 67.3472 (-0.0098) |
| multicore | higgs | c45_unpruned | 0.935 | 0.984 | +5.2% | no | 1317065/1317047 | 69.7672 / 69.8228 (+0.0556) |
| multicore | chess_1m | cart_depth10 | 0.017 | 0.017 | +4.8% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| multicore | chess_1m | c45_depth10_unpruned | 0.020 | 0.021 | +4.5% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| multicore | chess_10m | cart_depth10 | 0.049 | 0.049 | +0.8% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| multicore | chess_10m | c45_depth10_unpruned | 0.065 | 0.058 | -11.2% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| multicore | chess_100m | cart_depth10 | 0.247 | 0.261 | +5.8% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| multicore | chess_100m | c45_depth10_unpruned | 0.274 | 0.291 | +6.5% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| multicore | chess_1b | cart_depth10 | 1.824 | 1.967 | +7.8% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| multicore | chess_1b | c45_depth10_unpruned | 2.074 | 2.051 | -1.1% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| multicore | mnist8m | cart_depth10 | 3.331 | 3.363 | +0.9% | no | 2047/2047 | 84.5100 / 84.5200 (+0.0100) |
| multicore | mnist8m | c45_depth10_unpruned | 3.414 | 3.644 | +6.8% | no | 1453/1465 | 74.6500 / 74.6700 (+0.0200) |
| multicore | ecbdl_15m | cart_depth10 | 4.537 | 4.579 | +0.9% | no | 1863/1869 | 98.3076 / 98.2753 (-0.0323) |
| multicore | ecbdl_15m | c45_depth10_unpruned | 4.770 | 4.837 | +1.4% | no | 725/773 | 98.3161 / 98.3186 (+0.0025) |
| multicore | sorel_4m | cart_depth10 | 4.966 | 5.167 | +4.0% | no | 1619/1619 | 92.5749 / 92.5748 (-0.0001) |
| multicore | sorel_4m | c45_depth10_unpruned | 4.987 | 5.011 | +0.5% | no | 299/303 | 78.2971 / 78.2972 (+0.0001) |
| multicore | ecbdl | cart_depth10 | 9.266 | 9.362 | +1.0% | no | 1917/1927 | 98.3020 / 98.3161 (+0.0141) |
| multicore | ecbdl | c45_depth10_unpruned | 9.855 | 9.827 | -0.3% | no | 785/773 | 98.3171 / 98.3184 (+0.0013) |
| multicore | sorel | cart_depth10 | 15.309 | 15.481 | +1.1% | no | 1767/1767 | 92.6518 / 92.6492 (-0.0026) |
| multicore | sorel | c45_depth10_unpruned | 15.736 | 15.329 | -2.6% | no | 379/379 | 78.1126 / 78.1127 (+0.0001) |
| serial | covertype | cart_depth10 | 0.316 | 0.318 | +0.7% | yes | 1499/1499 | 77.6622 / 77.6622 (+0.0000) |
| serial | covertype | c45_depth10_unpruned | 0.339 | 0.340 | +0.2% | yes | 723/723 | 59.9327 / 59.9327 (+0.0000) |
| serial | covertype | cart_full | 0.650 | 0.653 | +0.5% | no | 48095/48233 | 93.8788 / 93.8702 (-0.0086) |
| serial | covertype | c45_unpruned | 0.826 | 0.837 | +1.3% | no | 30321/30335 | 94.4467 / 94.4450 (-0.0017) |
| serial | susy | cart_depth10 | 2.607 | 4.179 | +60.3% | no | 1959/1965 | 79.0082 / 79.0106 (+0.0024) |
| serial | susy | c45_depth10_unpruned | 2.487 | 3.091 | +24.3% | no | 743/769 | 78.6472 / 78.6490 (+0.0018) |
| serial | higgs | cart_depth10 | 7.876 | 9.524 | +20.9% | no | 2029/2029 | 70.5172 / 70.5136 (-0.0036) |
| serial | higgs | c45_depth10_unpruned | 7.446 | 7.847 | +5.4% | no | 935/927 | 63.7296 / 63.7340 (+0.0044) |
| serial | chess_1m | cart_depth10 | 0.019 | 0.021 | +8.4% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| serial | chess_1m | c45_depth10_unpruned | 0.026 | 0.027 | +6.2% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| serial | chess_10m | cart_depth10 | 0.225 | 0.213 | -5.5% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| serial | chess_10m | c45_depth10_unpruned | 0.267 | 0.251 | -6.1% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| serial | chess_100m | cart_depth10 | 2.082 | 2.019 | -3.0% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| serial | chess_100m | c45_depth10_unpruned | 2.432 | 2.336 | -3.9% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| serial | chess_1b | cart_depth10 | 39.343 | 38.499 | -2.1% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| serial | chess_1b | c45_depth10_unpruned | 42.329 | 41.218 | -2.6% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| serial | mnist8m | cart_depth10 | 115.178 | 115.645 | +0.4% | no | 2047/2047 | 84.5100 / 84.5200 (+0.0100) |
| serial | mnist8m | c45_depth10_unpruned | 113.768 | 113.500 | -0.2% | no | 1453/1465 | 74.6500 / 74.6700 (+0.0200) |
| serial | ecbdl_15m | cart_depth10 | 146.562 | 146.480 | -0.1% | no | 1863/1869 | 98.3076 / 98.2753 (-0.0323) |
| serial | ecbdl_15m | c45_depth10_unpruned | 150.093 | 144.096 | -4.0% | no | 725/773 | 98.3161 / 98.3186 (+0.0025) |
| serial | sorel_4m | cart_depth10 | 176.639 | 183.769 | +4.0% | no | 1619/1619 | 92.5749 / 92.5748 (-0.0001) |
| serial | sorel_4m | c45_depth10_unpruned | 174.417 | 168.279 | -3.5% | no | 299/303 | 78.2971 / 78.2972 (+0.0001) |
| gpu | covertype | cart_depth10 | 0.167 | 0.161 | -4.0% | yes | 1499/1499 | 77.6622 / 77.6622 (+0.0000) |
| gpu | covertype | c45_depth10_unpruned | 0.170 | 0.164 | -3.9% | yes | 723/723 | 59.9327 / 59.9327 (+0.0000) |
| gpu | covertype | cart_full | 0.235 | 0.211 | -9.9% | no | 48095/48249 | 93.8788 / 93.8684 (-0.0104) |
| gpu | covertype | c45_unpruned | 0.239 | 0.223 | -6.7% | no | 30321/30335 | 94.4467 / 94.4450 (-0.0017) |
| gpu | susy | cart_depth10 | 0.263 | 0.237 | -10.0% | no | 1959/1961 | 79.0082 / 79.0426 (+0.0344) |
| gpu | susy | c45_depth10_unpruned | 0.343 | 0.271 | -21.0% | no | 743/769 | 78.6472 / 78.6490 (+0.0018) |
| gpu | susy | cart_full | 1.356 | 27.244 | +1909.1% | no | 1194859/1442229 | 71.5684 / 71.6556 (+0.0872) |
| gpu | susy | c45_unpruned | 0.597 | 0.417 | -30.2% | no | 24683/24371 | 79.4952 / 79.5008 (+0.0056) |
| gpu | higgs | cart_depth10 | 0.508 | 0.470 | -7.5% | no | 2029/2029 | 70.5172 / 70.5170 (-0.0002) |
| gpu | higgs | c45_depth10_unpruned | 0.639 | 0.515 | -19.3% | no | 935/927 | 63.7296 / 63.7540 (+0.0244) |
| gpu | chess_1m | cart_depth10 | 0.147 | 0.136 | -7.5% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| gpu | chess_1m | c45_depth10_unpruned | 0.140 | 0.146 | +4.4% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| gpu | chess_10m | cart_depth10 | 0.150 | 0.152 | +1.3% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| gpu | chess_10m | c45_depth10_unpruned | 0.160 | 0.154 | -3.6% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| gpu | chess_100m | cart_depth10 | 0.414 | 0.412 | -0.6% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| gpu | chess_100m | c45_depth10_unpruned | 0.437 | 0.438 | +0.4% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| hybrid | covertype | cart_depth10 | 0.371 | 0.370 | -0.5% | yes | 1499/1499 | 77.6622 / 77.6622 (+0.0000) |
| hybrid | covertype | c45_depth10_unpruned | 0.366 | 0.360 | -1.7% | yes | 723/723 | 59.9327 / 59.9327 (+0.0000) |
| hybrid | covertype | cart_full | 0.428 | 0.423 | -1.2% | no | 48095/48249 | 93.8788 / 93.8684 (-0.0104) |
| hybrid | covertype | c45_unpruned | 0.420 | 0.423 | +0.6% | no | 30321/30335 | 94.4467 / 94.4450 (-0.0017) |
| hybrid | susy | cart_depth10 | 0.483 | 0.454 | -6.0% | no | 1959/1961 | 79.0082 / 79.0426 (+0.0344) |
| hybrid | susy | c45_depth10_unpruned | 0.538 | 0.490 | -8.9% | no | 743/769 | 78.6472 / 78.6490 (+0.0018) |
| hybrid | susy | cart_full | 0.928 | 26.845 | +2793.1% | no | 1194859/1442229 | 71.5684 / 71.6556 (+0.0872) |
| hybrid | susy | c45_unpruned | 0.785 | 0.627 | -20.2% | no | 24683/24371 | 79.4952 / 79.5008 (+0.0056) |
| hybrid | higgs | cart_depth10 | 0.726 | 0.679 | -6.4% | no | 2029/2029 | 70.5172 / 70.5170 (-0.0002) |
| hybrid | higgs | c45_depth10_unpruned | 0.841 | 0.730 | -13.2% | no | 935/927 | 63.7296 / 63.7540 (+0.0244) |
| hybrid | chess_1m | cart_depth10 | 0.326 | 0.327 | +0.1% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| hybrid | chess_1m | c45_depth10_unpruned | 0.333 | 0.326 | -2.0% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| hybrid | chess_10m | cart_depth10 | 0.351 | 0.351 | +0.2% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| hybrid | chess_10m | c45_depth10_unpruned | 0.359 | 0.357 | -0.6% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| hybrid | chess_100m | cart_depth10 | 0.641 | 0.635 | -0.9% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
| hybrid | chess_100m | c45_depth10_unpruned | 0.656 | 0.661 | +0.7% | yes | 17/17 | 100.0000 / 100.0000 (+0.0000) train |
