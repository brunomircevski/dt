# Benchmark results

One directory per benchmark: `<backend>-<machine>-<dataset>/`.

- **backend:** `cpu` or `cuda`.
- **machine:** a short name for the computer. `pc` is the i7-14700KF desktop (8 performance + 12 efficiency cores, 28 threads); `legion` is the laptop with an i7-13650HX (6 + 8 cores, 20 threads) and an RTX 5070 Laptop GPU (8 GB).
- **dataset:** the `bench/datasets.toml` entry, e.g. `susy500k` for `susy_500k`, `susy` for the full dataset.

Each directory holds a `run.sh` that runs that benchmark and writes all of its results into the same directory:

```bash
bench/results/cpu-pc-susy500k/run.sh          # each case once
bench/results/cpu-pc-susy500k/run.sh -m 5     # each case 5 times
```

`BENCHMARK.md` next to it describes the method, and the results once the run is done. `run.sh` refuses to run where results already exist, so move them away to run again.

| Directory | Status |
|---|---|
| `cpu-pc-susy500k` | run on 2026-10-09, one run per case |
| `cpu-pc-susy` | full SUSY (4.5M training rows), ready, not run yet |
| `cpu-pc-susy-scaling` | ./tree only, CART and C4.5 on full SUSY with 1–28 threads (scaling curve), ready, not run yet |
| `cuda-legion-susy` | ./tree only on the laptop, full SUSY: serial, parallel (20 threads) and CUDA, the protocols of `cpu-pc-susy`; run on 2026-10-10, one run per case; `machines.png` compares it with the desktop |

## A new machine

1. Copy a directory and rename it for the machine, e.g. `cpu-server-susy500k`.
2. In its `run.sh`, adjust `--cpus` to that machine's CPU numbering:
   - the first CPU listed runs the 1-thread cases and should be a fast core;
   - `--threads 1,all` uses every CPU listed.
3. Run `bench/setup.sh` once, prepare the data (`bench/.venv/bin/python bench/prepare.py susy_500k`), then run `run.sh`.

CUDA benchmarks get `cuda-<machine>-<dataset>` directories the same way (`cuda-legion-susy`): `bench/run.py --impls tree_cuda` runs `./tree --cuda` (built by `make tree`) with the largest thread count and also records the peak GPU memory.
