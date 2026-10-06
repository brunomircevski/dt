# Decision trees: CART and C4.5 on CPU and GPU

A learning project that grows **CART** and **C4.5** decision trees on numeric
data, with three interchangeable backends that produce the *same* tree:

| Backend | Flag | How |
|---------|------|-----|
| Serial | `--serial` | One thread. The reference. |
| Parallel | `--parallel` | Thread pool: features of a big node in parallel, subtrees as tasks. |
| Cuda | `--cuda` | GPU grows the large nodes level by level; small subtrees go to the CPU pool. |

* [ALGORITHMS.md](ALGORITHMS.md) — what exactly CART and C4.5 do here, how it
  was verified against scikit-learn and Quinlan's original C4.5, and what was
  wrong in the previous version.
* [CPU.md](CPU.md) — presorted columns and the parallel CPU builder.
* [CUDA.md](CUDA.md) — the GPU builder, kernel by kernel.

## Build

```bash
make          # ./tree      (needs CUDA: nvcc in /opt/cuda/bin, override with NVCC=...)
make cpu      # ./tree_cpu  (no CUDA needed; --cuda is unavailable)
```

## Run

```bash
./tree --cart datasets/covertype.csv                       # CART, Gini, full tree
./tree --cart --cv 10 --holdout 0.2 datasets/covertype.csv # CART pruned by 10-fold CV
./tree --c45 datasets/covertype.csv                        # C4.5 with its default pruning
./tree --c45 --cuda -d 20 datasets/supersymmetry.csv       # on the GPU, depth limit 20
./tree --help
```

The CSV needs a header; the last column is the class, a first column called
`Id` is ignored, every other column must be numeric. Default settings for a
run without arguments are at the top of `main.cpp`.

Useful flags: `--holdout F` (evaluate on a held-out fraction), `--print`,
`--dump FILE` (text tree), `--svg FILE` (rendered with
`python/render_tree_svg.py`), `--load-tree FILE` (prune an existing tree).

## Checks

```bash
tests/check_backends.sh       # serial, parallel and cuda grow identical trees
python3 python/compare_reference.py cart datasets/diabetes.csv           # vs scikit-learn
python3 python/compare_reference.py c45 datasets/diabetes.csv --c45 PATH # vs original c4.5
```

The original C4.5 (Release 8) source is available from Ross Quinlan's site
(`c4.5r8.tar.gz`). With a modern gcc, build it from `R8/Src` with the standard
headers forced in (the K&R code relies on implicit declarations, which silently
breaks `atof` and `-c`):

```bash
gcc -std=gnu89 -O2 -w -fcommon -Dcfree=free -include stdlib.h -include math.h -include string.h -include stdio.h -o c4.5 c4.5.c besttree.c build.c info.c discr.c contin.c subset.c prune.c stats.c st-thresh.c classify.c confmat.c sort.c getnames.c getdata.c trees.c getopt.c header.c -lm
```

## Performance

Tree-building time (`build`, best of 2) on a laptop: i7-13650HX (20 threads),
RTX 5070 Laptop running power-limited. Expect about ±15% run-to-run noise.
"Before" is the previous version of this project, measured on the same machine.

| Dataset | Algorithm | Serial | Parallel | Cuda | Before: serial / parallel / cuda |
|---------|-----------|-------:|---------:|-----:|------|
| covertype (581k × 54, 7 classes) | CART, depth 30 | 1.91 s | 0.25 s | 0.22 s | 39.8 s / 4.09 s / 1.39 s |
| covertype | CART, full tree | | 0.26 s | 0.21 s | |
| covertype | C4.5 | | 0.40 s | 0.28 s | |
| SUSY (5M × 18, 2 classes) | CART, depth 30 | | 1.62 s | 1.22 s | – / 31.8 s / 10.4 s |
| SUSY | CART, full tree (1.3M nodes) | | 1.72 s | 1.54 s | |
| SUSY | C4.5 | | 2.22 s | 1.07 s | |

C4.5 additionally spends 0.1–0.9 s on pruning and thresholds (`train total`).
Loading the 5M-row CSV takes about 0.25 s.

## Code map

| File | Role |
|------|------|
| `main.cpp` | Defaults, command line, reporting. |
| `options.h/.cpp` | All settings, grouped by algorithm and backend. |
| `dataset.h/.cpp` | Column-major dataset, parallel CSV loader. |
| `split_math.h` | Impurity / gain / tie rules shared by CPU and GPU. |
| `split_rules.h/.cpp` | CART vs C4.5: stopping, minimum child size, feature choice. |
| `cpu_builder.h/.cpp` | Presort + CPU tree growing (serial and parallel). |
| `gpu_builder.cu` | GPU tree growing. |
| `pruning.h/.cpp` | C4.5 collapse / thresholds / error-based pruning, CART cost-complexity pruning. |
| `trainer.h/.cpp` | Puts it together (backend, post-processing, cross-validation). |
| `tree.h/.cpp` | The tree, prediction, printing, reading. |
| `thread_pool.h/.cpp` | Work queue with deadlock-free nested `parallelFor`. |
