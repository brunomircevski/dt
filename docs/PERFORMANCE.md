# Performance

Earlier measurements of `./tree` on its own, moved here from the main README. For the comparison with scikit-learn, rpart, Weka J48 and YaDT, see [bench/README.md](../bench/README.md) and the runs in [bench/results/](../bench/results/).

## Serial, parallel and Cuda backends

Training time (`train total`: presort, GPU setup, build, cross-validation,
pruning; loading and evaluation excluded) on a laptop: i7-13650HX (20 threads),
RTX 5070 Laptop GPU (8 GB, power-limited). The laptop throttles under
sustained load, so expect ±10–15% run-to-run noise (up to 2× on long parallel
runs). Measured with the former `bench/bench.sh`; the new CPU benchmark is in
[bench/README.md](../bench/README.md).

| Dataset | Algorithm | Serial | Parallel | Cuda |
|---------|-----------|-------:|---------:|-----:|
| covertype (581k × 54, 7 classes) | CART, full tree | 1.23 s | 0.21 s | 0.15 s |
| covertype | CART + 10-fold CV | 11.6 s | 2.03 s | 1.44 s |
| covertype | C4.5 | 1.93 s | 0.41 s | 0.26 s |
| SUSY (5M × 18, 2 classes) | CART, full tree (1.3M nodes) | 19.8 s | 2.08 s | 2.02 s |
| SUSY | CART + 10-fold CV | | 51.2 s | 25.1 s |
| SUSY | C4.5 | 16.0 s | 4.42 s | 1.56 s |
| HIGGS (11M × 28, 2 classes) | CART, full tree (3M nodes) | 64.4 s | 7.06 s | 8.56 s |
| HIGGS | CART + 10-fold CV | | 158 s | 85.2 s |
| HIGGS | C4.5 | 74.9 s | 10.7 s | 7.62 s |

## Compared with the previous version

Same machine, same trees (`tests/check.sh`). Parallel and Cuda rows are
interleaved runs of both binaries (best of 3–4), the others single runs.

| Dataset | Algorithm | Backend | Before | After | Speed-up |
|---------|-----------|---------|-------:|------:|--------:|
| covertype | CART | serial | 2.14 s | 1.23 s | 1.7× |
| covertype | CART | parallel | 0.32 s | 0.21 s | 1.5× |
| covertype | CART | cuda | 0.24 s | 0.15 s | 1.6× |
| covertype | CART + 10-fold CV | serial | 22.7 s | 11.6 s | 2.0× |
| covertype | CART + 10-fold CV | parallel | 4.81 s | 2.03 s | 2.4× |
| covertype | CART + 10-fold CV | cuda | 6.62 s | 1.44 s | 4.6× |
| covertype | C4.5 | serial | 3.08 s | 1.93 s | 1.6× |
| covertype | C4.5 | parallel | 1.25 s | 0.41 s | 3.1× |
| covertype | C4.5 | cuda | 0.38 s | 0.26 s | 1.5× |
| SUSY | CART | serial / parallel / cuda | 19.6 / 2.05 / 2.09 s | 19.8 / 2.08 / 2.02 s | 1.0× |
| SUSY | CART + 10-fold CV | parallel | 82.5 s | 51.2 s | 1.6× |
| SUSY | CART + 10-fold CV | cuda | 47.7 s | 25.1 s | 1.9× |
| SUSY | C4.5 | serial / parallel / cuda | 16.3 / 5.07 / 1.65 s | 16.0 / 5.13 / 1.56 s | 1.0× |
| HIGGS | CART + 10-fold CV | parallel | 200 s | 158 s | 1.3× |
| HIGGS | CART + 10-fold CV | cuda | 140 s | 85.2 s | 1.6× |
| HIGGS | CART / C4.5 | all | | | ≈1.0× (within noise) |

Where it comes from: features that are constant in a node are dropped for its
whole subtree (covertype: 44 of 54 features are binary, about half of the
scanned data), cross-validation reuses the presorted columns / GPU buffers
and builds fold row lists without sorting, and the CPU partition no longer
needs a full-size buffer per thread. SUSY and HIGGS have only continuous
features, so a single tree is grown as fast as before. On data-center GPUs
(fast double precision) the Cuda backend scores every cut in one pass instead
of two (`--gpu-sweep`), which this laptop GPU cannot show.
