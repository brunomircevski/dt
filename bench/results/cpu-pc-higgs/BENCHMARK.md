# HIGGS on the desktop: ./tree parallel

**Status:** run on 2026-10-10 at 12:46, one run per case, commit abc0a22, `performance` governor, load average 0.36. Results and chart: [`cuda-legion-higgs`](../cuda-legion-higgs/BENCHMARK.md#results).

The desktop half of a laptop-vs-desktop comparison on HIGGS; the laptop half (parallel and CUDA) is [`cuda-legion-higgs`](../cuda-legion-higgs/BENCHMARK.md), which also holds the chart of both.

## Run it

```bash
bench/results/cpu-pc-higgs/run.sh          # each case once
bench/results/cpu-pc-higgs/run.sh -m 5     # each case 5 times
```

- **Time:** a few minutes, plus preparing HIGGS once (the 7.8 GB CSV into `bench/data/higgs/`).
- **Before:** commit the code, set the `performance` governor (`sudo cpupower frequency-set -g performance`), close browsers and chat apps.
- **What `run.sh` does:**
  1. Refuses to overwrite existing results.
  2. Builds `./tree_cpu` and prepares HIGGS if it is missing.
  3. Runs the benchmark:
     `bench/run.py higgs --protocols cart_alpha,c45 --impls tree --threads all --cpus 2,0,1,3-27 --reps <-m> --timeout 3600`
  4. Writes `report.md`, `summary.csv`, `runs.csv`, `chart.png`, next to `results.jsonl`, `machine.json`, `plan.json` and `run.log`.

## Method

The protocols and measurements of [`cpu-pc-susy`](../cpu-pc-susy/BENCHMARK.md), ./tree only, on HIGGS, with one backend:

| Backend | Command | Threads |
|---|---|---|
| Parallel | `./tree_cpu --parallel --threads 28` | 28 (all logical CPUs of the i7-14700KF) |

- **No serial run:** the comparison is parallel CPU vs CUDA only.
- **Protocols:** `cart_alpha` (CART, depth ≤ 30, cost-complexity pruning at α = 1e-5) and `c45` (C4.5, CF 0.25, min 2 rows, no depth limit).
- **Data:** HIGGS (Baldi, Sadowski, Whiteson 2014; UCI): 11,000,000 rows, 28 numeric features, 2 classes; the published split, the last 500,000 rows for testing, so 10,500,000 training rows.
- **Same data on both machines:** the two machines have differently formatted copies of `datasets/higgs.csv` (here no `Id` column and labels `0`/`1`; on the laptop an `Id` column and labels `background`/`signal`) with the same rows and values. Both labels sort to the same classes (`c0` = background), so the prepared files the tools read must have the same SHA-256 on both machines (`plan.json`); `bench/machines_chart.py` refuses to draw otherwise.
