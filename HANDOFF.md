# Handoff: `sorel_4m` on hopper (updated 2026-11-03)

Only this is left of the SOREL work of 2026-11-02. Done: the `--simple`
rerun of SOREL (epyc now 0.94–0.97× dell, as explained in
`docs/PERFORMANCE.md`), and `sorel_4m` on ampere and epyc. Results, the
two charts, `cluster-sorel/BENCHMARK.md`, the READMEs and the paper's 4M rows
(CPU cells) are filled in. Delete this file when hopper's runs are in.

The jobs run on mordor3 (`ssh mordor3`, user `bmircevski`, checkout `~/dt`).

## Waiting

| Job | Node | What | Writes to `~/dt/bench/results/` |
|---|---|---|---|
| 19 `dt-cuda-hopper-sorel_4m` | hopper | CUDA hybrid, 10 runs | `cluster-sorel/cuda-hopper-sorel_4m/` |
| 22 `dt-gpuonly-hopper-sorel_4m` | hopper | `--gpu-only`, 10 runs | `cluster-sorel/gpuonly-hopper-sorel_4m/` |

Build `~/dt/tree` at 4ece22f (`src/` = da1f100), data in
`bench/data/sorel_4m/` (prepared by job 18), `--test-accuracy`, protocols
`cart_depth10,c45_depth10_unpruned`.

**Hopper is busy:** another user (ibalyka) holds all 4 GPUs with jobs 9–12
(5-day limit, started 2026-11-01 ~13:00, so up to ~2026-11-06). Ours wait in
the queue with `--exclusive` and start on their own. Don't cancel the other
user's jobs or `srun` there.

```bash
ssh mordor3 'bash -lc "squeue -a -u bmircevski; squeue -a -w hopper"'
ssh mordor3 'cd ~/dt; tail -n 3 bench/results/cluster-sorel/*hopper-sorel_4m/run.log'
```

## Finishing

1. `rsync -a mordor3:dt/bench/results/cluster-sorel bench/results/`
2. Check: every run `ok`, nodes 1,619 (CART) / 299 (C4.5), depth 10, test
   accuracy 92.57% / 78.30% (as ampere and epyc), VRAM peak ≈ 76 GiB, no run
   over twice the mean of the others.
3. `bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-sorel`
   (and `--table`).
4. Docs: in `bench/results/cluster-sorel/BENCHMARK.md` the hopper rows of the
   sorel_4m table, the "4M sample on the GPUs" bullet (H100 speed-up over
   epyc's 6.43 s / 7.85 s) and the "Hopper's runs are queued" sentence in
   **Runs**; `bench/results/README.md` (cluster-sorel row, "hopper queued").
   `paper/main.tex`: the `\est` GPU cells of both "4M sample" rows
   (\ImplGpu = gpuonly-hopper, \ImplHybrid = cuda-hopper; mean and min–max of
   `train_seconds`), then rebuild the PDF (`latexmk -pdf` in `paper/`).
5. Commit with explicit paths (another session may edit `paper/main.tex`)
   and delete this file. Push only if the user asks.

## Rules from the user

- Build, run and benchmark only on mordor3 compute nodes, never locally.
- No CPU benchmarks on hopper or ampere; don't `srun` into busy nodes.
- Charts show means only.
- Commit or push only when asked.
