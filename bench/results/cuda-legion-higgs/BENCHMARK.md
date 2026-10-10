# HIGGS on the Legion laptop: ./tree parallel and CUDA

**Status:** designed 2026-10-10, not run yet. The results will be written into this directory.

The laptop half of a laptop-vs-desktop comparison on HIGGS; the desktop half is [`cpu-pc-higgs`](../cpu-pc-higgs/BENCHMARK.md). Both are fresh runs; nothing is copied.

## Run it

```bash
bench/results/cuda-legion-higgs/run.sh          # each case once
bench/results/cuda-legion-higgs/run.sh -m 5     # each case 5 times
```

- **Time:** a few minutes, plus preparing HIGGS once.
- **Before:** the same as [`cuda-legion-susy`](../cuda-legion-susy/BENCHMARK.md): commit the code, AC power, platform profile `max-power`, `performance` governor, nothing else on the GPU.
- **What `run.sh` does:** as in `cuda-legion-susy`, with
  `bench/run.py higgs --protocols cart_alpha,c45 --impls tree,tree_cuda --threads all --cpus 4,0-3,5-19 --reps <-m> --timeout 3600`
- **Laptop vs desktop chart** (`machines.png`), once both runs are done; it refuses to draw if the data files or trees differ:
  ```bash
  bench/.venv/bin/python bench/machines_chart.py bench/results/cpu-pc-higgs \
      bench/results/cuda-legion-higgs --out bench/results/cuda-legion-higgs/machines.png
  ```

## Method

The protocols, data and measurements of [`cpu-pc-higgs`](../cpu-pc-higgs/BENCHMARK.md) (where the HIGGS data and its check are described), on the laptop, with two backends:

| Backend | Command | Threads |
|---|---|---|
| Parallel | `./tree_cpu --parallel --threads 20` | 20 (all) |
| CUDA | `./tree --cuda --threads 20` | GPU, plus 20 CPU threads for the small subtrees |

- **GPU memory:** ./tree keeps two copies of all sorted columns on the GPU: 10.5M rows × 28 features × 8 bytes × 2 ≈ 4.4 GiB, plus the CUDA context; the RTX 5070 Laptop GPU has 8 GB. ./tree checks the free memory before allocating and stops with an error if it does not fit.
- **Memory measured:** peak RSS up to the end of training; for CUDA also the peak GPU memory above idle (`nvidia-smi` every 10 ms).
