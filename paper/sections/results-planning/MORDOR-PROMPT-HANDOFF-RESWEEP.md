# Prompt for the agent on mordor3: re-sweep --handoff-rows for the deep trees, rerun only what changes

You work in the dt repository on the mordor3 SLURM cluster (build and run only on compute nodes, never on the login node; one exclusive job per GPU node). Commit and push results when done (GitHub push works there). Do not rerun anything whose hand-off value does not change.

## Why
The paper states that Hybrid always runs at the best `--handoff-rows` for its GPU node, dataset and depth limit. That is not true at depth 15–30.
- `bench/tools/handoff/handoff_rows.json` holds only `depth10` (all five GPU datasets, both nodes; sweep of 2026-11-16) and `nodepth` (HIGGS, MNIST8M, SOREL 4M; sweep of 2026-11-15/16).
- `run.py::handoff_rows` gives depths 15–30 the `nodepth` value: it takes the shallowest stored limit at least as deep as the run. Depths 15, 20, 25 and 30 were never swept.
- Example: HIGGS depth 30 on hopper uses the `nodepth` pick 256 and takes 0.847 s (`bench/hybrid-vs-gpu-only/cuda-hopper-higgs`). The default 512 took 0.821 s in `bench/cluster-gpu-vs-others/one-tree/vs-hopper-higgs-depth30`, because `final.sh` forces `HANDOFF=default` for that line.
- Both sweeps ran before ./tree 7f98876f ("less RAM and VRAM for --hybrid and --gpu, same trees and speed"). The final GPU results are on 7f98876f.
- The hand-off buffers are fixed in size (2 × 128 MB, `src/build/gpu_host.cuh` `kHandOffBytes`), so the timed set-up does not depend on the value. Sweeping without the set-up, as `handoff_sweep.sh` does, stays valid.

## What is already good (do not redo)
- **The `depth10` picks:** at depth 10 the hybrid hands off almost nothing, and the depth-10 sweeps are flat (all values within about 1–2 %; see `bench/tools/handoff/depth10/*.json`). Keep them. Keep every depth-5 and depth-10 Hybrid result that uses them: `cluster-<ds>/cuda-*`, the `hybrid-vs-gpu-only` depth 5/10 points, `cluster-gpu-vs-others` d10, the cluster-power ECBDL'14 rows and the profiling runs.
- **Every `--gpu` (GPU-only) and CPU result:** the value does not affect them.

## Steps
1. **Check the build.** Make sure the build you sweep is the one the final results used: 7f98876f, or a later commit with no change under `src/` since. Put `TREE_GIT_DESCRIBE` in every job.
2. **Confirm `nodepth` on the final build.** Use `bench/tools/handoff/handoff_sweep.sh` with `DEPTH=none` and `SWEEP_RUNS=5`, on hopper and on ampere, for HIGGS, MNIST8M and SOREL 4M. Limit `VALUES` to the current pick and its two neighbours on each side by factors of 2, plus `default`:
   - HIGGS: 256 for hopper, 512 for ampere;
   - MNIST8M: 8192;
   - SOREL 4M: 16384.

   Write into a new results folder (e.g. `bench/tools/handoff/nodepth-7f98876f/`) so the old sweep stays in git.
3. **Sweep depths 15, 20, 25 and 30.** Use `DEPTH=15|20|25|30`, the same three datasets and both nodes, with `VALUES="default 256 512 1024 2048 4096 8192 16384 32768"`.
   - For SOREL 4M you may drop 256–1024, since its curve falls steadily up to 16384 (see `bench/hybrid-vs-gpu-only/BENCHMARK.md`, "the sweep").
   - Results go to `bench/tools/handoff/depth<N>/`. `pick_handoff.py` writes the `depth<N>` keys into `handoff_rows.json`. `run.py` already looks up any `depth<N>` key.
   - Estimate the job time before submitting, so that each node's job stays within a few hours.
4. **Settle close calls.** If the best two values of a case are within 2 % of each other, rerun just those two with 10 interleaved runs each and keep the faster one.
5. **List what changed.** Compare the new picks with the values the existing results used, for each (node, dataset, depth). Write the table into `bench/tools/handoff/README` or a new `bench/tools/handoff/RESWEEP-2026-11.md`:

   | node | dataset | depth | value used | new pick | sweep times |
   |---|---|---|---|---|---|
6. **Rerun only the cases whose value changed,** with 10 runs each, the same protocols and the same harness:
   - **`hybrid-vs-gpu-only`:** only the `cuda-<node>-<dataset>` points at the changed depths. If `hybrid_vs_gpu_only.sh` can't rerun a subset of depths, add an option for that rather than rerunning everything. Keep GPU-only as it is.
   - **`cluster-gpu-vs-others/one-tree/vs-hopper-higgs-depth30`:**
     - remove `HANDOFF=default` from `bench/tools/cluster/final.sh` (the hybrid takes the tuned `depth30` value);
     - rerun only the `tree_cuda` runs if the harness allows it, otherwise the whole line, which is cheap (HIGGS);
     - note the change in its `BENCHMARK.md`.
   - **If and only if a `nodepth` pick changed in step 2:**
     - the Hybrid no-depth-limit points;
     - `bench/cluster-power/mnist8m-nodepth`, Hybrid rows only (hopper);
     - `bench/hybrid-vs-gpu-only/profile/cuda-*-mnist8m`.
7. **Update the docs and charts:**
   - regenerate the charts in the touched `BENCHMARK.md`s (`hybrid-vs-gpu-only` both nodes, `cluster-gpu-vs-others`);
   - update their tables and the "When" lines (build, jobs, dates);
   - in the paper, check `paper/sections/experiment-design.tex` §4.3: it must say the value is the fastest of a sweep for each dataset, GPU server and depth limit;
   - list every number in `paper/sections/*.tex` that changed (old → new), but do not rewrite prose.
8. **Report back:**
   - the table from step 5;
   - the reruns, with job IDs;
   - the old → new numbers, especially HIGGS depth 30 on hopper (Hybrid vs GPU-only vs cuML/XGBoost);
   - the commit hash.

## Out of scope
- cuML and XGBoost stay on hopper only (author decision 2026-11-24). Do not add ampere lines to `final.sh -k gpulibs`.
- Memory sizes in the paper are now in GiB (1 GiB = 2^30 bytes). If you list paper numbers in step 7, use GiB.
