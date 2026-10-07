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
* [docs/PYTHON_COMPARISON.md](docs/PYTHON_COMPARISON.md) — speed and accuracy
  against scikit-learn and the Python C4.5 packages.

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
./tree --cart datasets/covertype.csv                # CART as in Breiman et al.: Gini, alpha chosen on a test sample
./tree --cart --cv 10 datasets/covertype.csv        # alpha by 10-fold cross-validation (11 trees)
./tree --cart --no-prune datasets/covertype.csv     # only grow the maximal tree
./tree --cart --alpha 1e-5 datasets/covertype.csv   # prune with a fixed alpha
./tree --c45 datasets/covertype.csv                 # C4.5 as in Quinlan's c4.5: -m 2, CF 0.25
./tree --c45 --cuda --holdout 0.2 datasets/higgs.csv
./tree --help
```

The CSV needs a header; the last column is the class, a first column called
`Id` is ignored, every other column must be numeric.

Defaults follow the published algorithms: CART as described by Breiman et al.
(1984) and C4.5 as in Quinlan's program (Release 8). Defaults live in
`src/core/options.h`.

### Options

A flag that the chosen algorithm or backend would ignore (e.g. `--cv` with
`--c45`, `--gpu-min-rows` without `--cuda`) or that contradicts another one
(e.g. `--cv` with `--alpha`, `--cf` with `--no-prune`) is an error.

**Backend and algorithm**

| Flag | Default | Meaning |
|------|---------|---------|
| `--serial` / `--parallel` / `--cuda` | `--parallel` | Where the tree is grown. All three grow the same tree. |
| `--cart` / `--c45` | `--cart` | Algorithm. |
| `--threads N` | 0 = all cores | CPU threads for `--parallel`, the CPU part of `--cuda`, loading and evaluation. |

**Pre-pruning** (stop growing early; the same two rules for both algorithms)

| Flag | Default | Meaning |
|------|---------|---------|
| `-d N`, `--max-depth N` | -1 = unlimited | Nodes at depth N become leaves. |
| `--min-leaf N` | CART 1, C4.5 2 | A split is only allowed if both children keep at least N rows. For C4.5 this is the c4.5 program's `-m`; C4.5 also raises the limit by itself on big nodes (to 10% of the average class size, at most 25), as the original does. |

**Post-pruning** (grow the full tree, then cut branches back; on by default)

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-prune` | pruning on | Keep the grown tree: no CART cost-complexity pruning (and no CV), no C4.5 error-based pruning. |

**Other**

| Flag | Default | Meaning |
|------|---------|---------|
| `--holdout F` | 0 | Put a random fraction F of the rows aside, train on the rest and report the accuracy on the put-aside rows (`test accuracy`). Tested once, on one split. |
| `--seed N` | 1 | Random seed for the holdout split and for CART's test sample or CV folds. The same seed gives the same split and folds on every machine and backend; change it to see how much the result depends on the split. |
| `-m N` | 1 | Duplicate the rows N times in memory (slightly rescaled) to stress-test with bigger data. |
| `--print` | off | Print the tree. |
| `--dump FILE` | off | Write the tree as text; `tools/render_tree_svg.py FILE out.svg` draws it. |

**CART post-pruning** (splits always use the Gini index, as in Breiman et al.)

| Flag | Default | Meaning |
|------|---------|---------|
| `--test-sample F` | 1/3 | Default. Grow the tree on 1 − F of the rows and choose alpha with the other F (see below). One tree. |
| `--cv K` | (off) | Choose alpha by K-fold cross-validation instead: the tree is grown on all rows, plus K fold trees. |
| `--alpha X` | (off) | Prune with this fixed alpha instead (one tree on all rows). Larger prunes more: each leaf must classify about X × rows more training rows correctly to be kept (covertype: `1e-6` 44k nodes, `1e-5` 10k, `1e-4` 0.9k). |

**C4.5 post-pruning**

| Flag | Default | Meaning |
|------|---------|---------|
| `--cf X` | 0.25 | Confidence factor of error-based pruning (the c4.5 program's `-c 25`); smaller prunes more. |

Subtree raising is always on, as in the original program.

**Performance tuning** (the tree does not change; only the speed)

| Flag | Default | Meaning |
|------|---------|---------|
| `--task-rows N` | 4096 | `--parallel` / `--cuda` CPU side: nodes with at least N rows grow a child as a separate pool task. |
| `--feature-parallel-rows N` | 65536 | Nodes with at least N rows scan and partition their features in parallel. |
| `--gpu-min-rows N` | 512 | `--cuda`: nodes with fewer rows are finished by the CPU pool while the GPU grows the big ones. |
| `--gpu-sweep auto\|one-pass\|two-pass` | `auto` | `--cuda`: score every cut in double precision (`one-pass`), or filter in single precision first and use double precision only for the candidates (`two-pass`). `auto` picks one-pass on GPUs with fast double precision (A100, H100, B200…). |
| `--gpu-profile` | off | Print per-level and per-kernel GPU times. Synchronises after every kernel, so it slows the build; do not use it for timing. |

The best values depend on the hardware and the data (`docs/CPU.md`,
`docs/CUDA.md` explain what each threshold does).

### How CART chooses its pruning strength

A grown CART tree fits the training data too closely, so CART prunes it.
Pruning has one parameter, alpha: the higher it is, the smaller the tree. CART
first lists every alpha at which the full tree would lose a branch (the
"pruning sequence", a series of ever smaller trees), then measures each of
those trees on rows they were not grown on, and keeps the smallest tree whose
error is within one standard error of the lowest (Breiman's 1-SE rule: near the
minimum the differences are noise, so the simpler tree wins). Breiman et al.
give two ways to get rows the tree has not seen:

* **Test sample** (default, `--test-sample F`): put a random fraction F of the
  rows aside (`--seed`), grow the tree on the rest, measure the pruning
  sequence on the rows put aside. One tree, so it costs about as much as C4.5's
  pruning. The tree is grown on fewer rows (2/3 by default); on big datasets a
  smaller F such as 0.1 is usually enough to measure the error and leaves more
  rows for growing.
* **Cross-validation** (`--cv K`, better for small datasets): grow
  the tree on all rows; then split the rows into K parts, grow K more trees,
  each on all parts but one, and measure each on its left-out part. Every row
  is used for growing and for measuring, but it costs K + 1 trees.

On covertype and SUSY with `--holdout 0.2` (the accuracy is on rows used for
neither growing nor pruning):

| Run | Nodes | Test accuracy | train total |
|-----|------:|--------------:|------------:|
| covertype CART `--no-prune` | 48,089 | 93.85% | 0.19 s |
| covertype CART (test sample 1/3) | 28,685 | 92.46% | 0.16 s |
| covertype CART `--test-sample 0.1` | 33,807 | 93.60% | 0.19 s |
| covertype CART `--cv 10` | 24,797 | 93.67% | 1.70 s |
| covertype C4.5 | 26,631 | 94.20% | 0.37 s |
| SUSY CART `--no-prune` | 1,063,237 | 71.64% | 2.38 s |
| SUSY CART (test sample 1/3) | 1,489 | 79.65% | 1.30 s |
| SUSY C4.5 | 13,889 | 79.63% | 1.80 s |

### Timings

Every run prints:

| Line | What it measures |
|------|------------------|
| `load` | Reading the CSV (not part of training). |
| `gpu setup` | `--cuda`: allocating and uploading to the GPU. |
| `presort` | Sorting every feature column once (needed by all backends). |
| `build` | Growing the tree (CART with a test sample: on the rows not put aside). |
| `cross-validation` | CART `--cv` only: growing and scoring the fold trees. |
| `prune` | Pruning. CART: putting the test sample aside, measuring the pruning sequence on it, pruning. C4.5: its threshold and collapse passes and error-based pruning. |
| `train total` | Sum of the training lines above. |
| `evaluate` | Computing the accuracies (not part of training). |

### Comparing CART and C4.5 timings

With default settings both algorithms do the same steps: presort, grow one
tree, prune it. `train total` and each line can be compared directly. Keep in
mind that default CART grows its tree on 2/3 of the rows (the rest is its test
sample) while C4.5 grows on all of them; `--no-prune` grows both on all rows.

With `--cv K`, CART also grows K fold trees, reported on the `cross-validation`
line. A CV run prints the alpha it chose at full precision:

```
  same tree without CV: --alpha 1.7211348474730298e-06
```

and running with that `--alpha` grows exactly the same tree without the
cross-validation (the CV result is the same on every machine for the same
`--seed`).

## Tests and benchmarks

```bash
make test                     # every backend must grow the verified trees in tests/golden/
TREE_BIN=./tree_cpu tests/check.sh
bench/bench.sh                # timing table for the datasets found in datasets/
python3 tools/compare_reference.py cart datasets/diabetes.csv           # vs scikit-learn
python3 tools/compare_reference.py c45 datasets/diabetes.csv --c45 PATH # vs original c4.5
python3 bench/compare_python.py prep && python3 bench/compare_python.py run covertype_20k  # vs Python libraries (docs/PYTHON_COMPARISON.md)
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
| covertype | CART + 10-fold CV | 11.6 s | 2.03 s | 1.44 s |
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
