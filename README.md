# Decision trees: CART and C4.5 on CPU and GPU

Grows **CART** and **C4.5** decision trees on numeric data with three
interchangeable backends that produce the *same* tree (byte for byte):

| Backend | Flag | How |
|---------|------|-----|
| Serial | `--serial` | One thread. The reference. |
| Parallel | `--parallel` | Thread pool: features of a big node in parallel, subtrees as tasks. |
| Cuda | `--cuda` | GPU grows the large nodes level by level; small subtrees go to the CPU pool. |

* [docs/ALGORITHMS.md](docs/ALGORITHMS.md) — what exactly CART and C4.5 do
  here and how it was verified against scikit-learn and Quinlan's C4.5.
* [docs/CPU.md](docs/CPU.md) — presorted columns and the CPU builder.
* [docs/CUDA.md](docs/CUDA.md) — the GPU builder, kernel by kernel.

## Build

```bash
make          # ./tree      (CPU + CUDA; nvcc in /opt/cuda/bin, override with NVCC=..., CUDA_LIB=...)
make cpu      # ./tree_cpu  (no CUDA needed; --cuda is unavailable)
```

`make` compiles the kernels for the GPU of the build machine. For other GPUs
(e.g. a cluster with A100 and H100):

```bash
make CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90"
```

## Run

```bash
./tree --cart datasets/covertype.csv                # CART: Gini, alpha by 10-fold CV (Breiman)
./tree --cart --no-prune datasets/covertype.csv     # the maximal tree
./tree --cart --alpha 1e-5 datasets/covertype.csv   # fixed cost-complexity alpha
./tree --c45 datasets/covertype.csv                 # C4.5 with its default pruning (CF 0.25)
./tree --c45 --cuda --holdout 0.2 datasets/higgs.csv
./tree --help
```

The CSV needs a header; the last column is the class, a first column called
`Id` is ignored, every other column must be numeric. Pruning is on by default
for both algorithms (`--no-prune` turns it off). Defaults are in
`src/core/options.h`.

Useful flags: `--holdout F` (evaluate on a held-out fraction), `--threads N`,
`--print`, `--dump FILE` (text tree; `tools/render_tree_svg.py FILE out.svg`
draws it), `-m N` (duplicate the rows N times for stress tests),
`--gpu-min-rows N`, `--gpu-sweep auto|one-pass|two-pass`.
`DT_GPU_VERBOSE=1` prints per-level and per-kernel GPU times.

## Tests and benchmarks

```bash
make test                     # every backend must grow the verified trees in tests/golden/
TREE_BIN=./tree_cpu tests/check.sh
bench/bench.sh                # timing table for the datasets found in datasets/
python3 tools/compare_reference.py cart datasets/diabetes.csv           # vs scikit-learn
python3 tools/compare_reference.py c45 datasets/diabetes.csv --c45 PATH # vs original c4.5
```

`tests/golden/` holds the trees of the version that was checked node by node
against scikit-learn and Quinlan's C4.5 (iris and diabetes are in
`tests/data/`; covertype and SUSY samples are used when `datasets/` has them).

The original C4.5 (Release 8) source is available from Ross Quinlan's site
(`c4.5r8.tar.gz`). With a modern gcc, build it from `R8/Src` with the standard
headers forced in (the K&R code relies on implicit declarations, which silently
breaks `atof` and `-c`):

```bash
gcc -std=gnu89 -O2 -w -fcommon -Dcfree=free -include stdlib.h -include math.h -include string.h -include stdio.h -o c4.5 c4.5.c besttree.c build.c info.c discr.c contin.c subset.c prune.c stats.c st-thresh.c classify.c confmat.c sort.c getnames.c getdata.c trees.c getopt.c header.c -lm
```

## Performance

Training time (`train total`: presort, GPU setup, build, cross-validation,
pruning; loading and evaluation excluded) on a laptop: i7-13650HX (20 threads),
RTX 5070 Laptop GPU (8 GB, power-limited). The laptop throttles under
sustained load, so expect ±10–15% run-to-run noise (up to 2× on long parallel
runs); `bench/bench.sh` regenerates the table.

| Dataset | Algorithm | Serial | Parallel | Cuda |
|---------|-----------|-------:|---------:|-----:|
| covertype (581k × 54, 7 classes) | CART, full tree | 1.23 s | 0.21 s | 0.15 s |
| covertype | CART + 10-fold CV (default) | 11.6 s | 2.03 s | 1.44 s |
| covertype | C4.5 | 1.93 s | 0.41 s | 0.26 s |
| SUSY (5M × 18, 2 classes) | CART, full tree (1.3M nodes) | 19.8 s | 2.08 s | 2.02 s |
| SUSY | CART + 10-fold CV | | 51.2 s | 25.1 s |
| SUSY | C4.5 | 16.0 s | 4.42 s | 1.56 s |
| HIGGS (11M × 28, 2 classes) | CART, full tree (3M nodes) | 64.4 s | 7.06 s | 8.56 s |
| HIGGS | CART + 10-fold CV | | 158 s | 85.2 s |
| HIGGS | C4.5 | 74.9 s | 10.7 s | 7.62 s |

### Compared with the previous version

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

## Code map

| Path | Role |
|------|------|
| `src/app/main.cpp` | Command line, reporting. |
| `src/app/trainer.*` | Puts it together: backend, CART cross-validation, post-processing. |
| `src/core/options.*` | All settings, grouped by algorithm and backend. |
| `src/core/dataset.*` | Column-major dataset, parallel CSV loader. |
| `src/core/tree.*` | The tree (flat arrays in preorder), prediction, printing. |
| `src/core/thread_pool.*` | Work queue with deadlock-free nested `parallelFor`. |
| `src/algo/split_math.h` | Impurity / gain / tie rules shared by CPU and GPU. |
| `src/algo/split_rules.*` | CART vs C4.5: stopping, minimum child size, feature choice. |
| `src/algo/cart_pruning.*` | Cost-complexity pruning and the weakest-link sequence. |
| `src/algo/c45_pruning.*` | C4.5 collapse, training-value thresholds, error-based pruning. |
| `src/build/grower.h` | Backend interface: grow a tree on (a subset of) the training rows. |
| `src/build/cpu_builder.*`, `cpu_grower.cpp` | Presort + CPU tree growing (serial and parallel). |
| `src/build/gpu_grower.cu` | GPU tree growing. |
| `src/build/node_store.*` | Thread-safe node storage used while growing. |
| `tests/`, `bench/`, `tools/` | Golden-tree tests, benchmark script, reference comparison and SVG rendering. |
