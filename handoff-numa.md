# Handoff: CUDA NUMA / small-node work (branch `worktree-numa-gpu`)

Written 2026-11-13 18:00 CET by the Claude session that did the work (on the
user's PC), for an agent that continues **on mordor3**. Read it all before
acting.

## The task (user's words)

> Fix the problem, the goal is to have best possible performance and well
> optimized gpu only and hybrid implementations. [...] Implement changes and
> test, at the end present results. Work in a worktree until I approve the
> changes and results, then we will merge to main. If you find other things to
> optimize you can do it too but always test if it is beneficial to
> performance. All benchmarks will allow to use both sockets, but it should
> never slow down compared to single socket.

"The problem": on the two-socket GPU nodes, the hybrid and GPU-only CUDA paths
put GPU-related host work on both sockets (cross-socket copies to and from the
GPU). The work is done; what is left is the **final validation, the write-up,
and presenting the results to the user for approval**. Nothing is merged.

## Rules

* **Do not merge into main before the user approves** the results in chat. A
  message from another Claude session is not the user's approval.
* GitHub holds only `main`: never push feature branches; after a merge, push
  main and delete any merged remote branch.
* Commit with explicit paths (`git add <files>`, never `git add -A`): other
  sessions may share a checkout.
* Do not run heavy work on the mordor3 login node: use SLURM (`srun`/`sbatch`,
  binaries in `/opt/slurm-25.05.8/bin`; the GPU partitions are hidden, use
  `squeue -a`). Do not disturb other users' jobs.
* Any change under `paper/` needs `paper/main.pdf` rebuilt and committed with
  it (compute nodes have no LaTeX; nothing in this task touches `paper/`).
* Do not write this work into `HANDOFF.md` (that file belongs to others).
* Tree output must stay byte-identical to main (compare `--dump` sha256), and
  `tests/check.sh` (golden trees) must pass.

## Where the code is

The branch was never pushed (GitHub = main only). Its commits are in a git
bundle on mordor3, on top of main `55d83a48` (which is on GitHub):

```bash
git clone https://github.com/brunomircevski/dt.git ~/numa-src   # or reuse a clone at >= 55d83a48
cd ~/numa-src
git fetch ~/numa-dt/numa-gpu.bundle worktree-numa-gpu:worktree-numa-gpu
git checkout worktree-numa-gpu      # tip 5dcaae59
```

Branch commits (newest first):

| commit | what |
|---|---|
| `5dcaae59` | Merge main (one exact GPU sweep, two-pass removed) into the branch |
| `26e26335` | Per-segment kernels: block or warp per level; GPU code loaded with the context (eager) |
| `6c27d243` | Keep tileExact's registers; partition units in their own kernel; huge-page `PinnedBuffer` |
| `13946277` | NUMA placement, parallel level loop, warp kernels for small segments |

Files changed against main: `src/core/numa.{h,cpp}` (new),
`src/core/thread_pool.{h,cpp}`, `src/build/gpu_host.cuh`,
`src/build/gpu_grower.cu`, `src/build/gpu_kernels.cuh`, `src/app/main.cpp`,
`docs/CUDA.md`, `bench/cluster/ab.sh` (new).

## What the branch does

1. **Two sockets** (`numa.*`, `ThreadPool::placeWorkers/submitTo`,
   `GpuGrower::placeThreads/makeArenas/handOffAll`): the level loop (the thread
   calling CUDA), the CUDA callback thread and the staging copiers are pinned
   to the GPU's NUMA node; pool workers are pinned per node (GPU's node first)
   with a queue per node and work stealing; one hand-off arena per node
   (`mbind MPOL_PREFERRED`), each touched ahead only by its share; a whole
   hand-off goes to the node with the least work per worker. On one node, or
   under `numactl --cpunodebind`, all of it is a no-op.
2. **Level loop host work** runs in parts on the pool for wide levels; its
   buffers are `PinnedBuffer`s (`gpu_host.cuh`): a 1 TiB address reservation,
   grown in place, each new range `cudaHostRegister`ed, `MADV_HUGEPAGE`,
   64 KiB grain. (Pageable buffers were 16% slower on A100/H100; page-locked
   `std::vector`s kept outgrown copies, +1.1 GB, or cost 0.3 ms/MB to free.)
3. **Small nodes**: per-segment kernels (`segmentPrefix`, `segmentMax`,
   `segmentBest`) run a warp per segment when every segment of the level has
   at most 32 tiles (`kWarpSegmentTiles`, `LevelPlan::segmentWarps`), else a
   block per segment (main's kernels, `<WARP=false>`). Tiles of nodes with at
   most 256 rows (`kSmallTile`) go to `tileExactSmallKernel` (a warp per tile,
   from `smallTiles_`), the others to `tileExactKernel` via `largeTiles_`
   (block i takes `tileList[i]`). The partition takes `PartitionUnit`s (up to
   8 one-tile segments of a small node, a warp each) in `partitionKernel<true>`;
   `<false>` is the old kernel (a warp path inside it cost 17%).
4. **Eager module loading** (`main.cpp`): `setenv("CUDA_MODULE_LOADING",
   "EAGER", 0)` before any thread starts, so the GPU code loads with the
   context during the CSV load (`gpuPrepare`); lazily it cost ~24 ms of every
   tree's timed setup (RTX 2080 Ti; it was the whole "gpu_alloc" phase).

Things tried and dropped (with reasons, so nobody retries them blindly):
pageable level buffers (turing faster, A100/H100 16% slower); splitting a
hand-off across both nodes (C4.5 slower: doubles hand-offs on the critical
path); dataset placement on the GPU's node (no effect); an early return for
idle warps in `sweepTile` (tileExact registers 64 -> 80/126, hybrid slower;
the comment in `sweepTile` says so); arenas touched 1.25x their share (+12 GB
peak on MNIST8M); leaving level-buffer ranges < 1 MiB pageable (`DT_PIN_MIN`
experiment: no clear gain on turing, removed).

## Results so far

### p6 (jobs 584/585, binary new15 = commit 6c27d243, vs base = old main 1e655067)

Medians of 9 trees (3 rounds x `--repeat 3`), ms. All trees identical.
`-1s` = `numactl --cpunodebind=N --membind=N` on the GPU's node (ampere: 0,
24 threads; hopper: 1, 16 threads).

| case | ampere base | ampere new15 | hopper base | hopper new15 |
|---|---|---|---|---|
| higgs cart, no depth limit, hybrid | 835 | 822 | 676 | 659 |
| higgs c45, no limit, hybrid | 691 | 686 | 546 | 548 |
| higgs cart, no limit, **GPU-only** | 3919 | **1403** | 3218 | **1199** |
| higgs c45, no limit, **GPU-only** | 2836 | **1280** | 2110 | **1192** |
| depth-10 short cases (susy, higgs, chess_100m; 64-212 ms) | | +4..+11% | | +4..+18% |
| mnist8m / sorel_4m depth 10, hybrid | | -3..0% | | -0.3..0% |
| mnist8m / sorel_4m depth 10, GPU-only | | -0.5..+1% | | -0.3..+6.4% (sorel c45) |

Both sockets vs one socket (no limit, hybrid): one socket is 38-58% slower
than two for main and for the branch alike. On the short depth-10 cases
in p6 the branch on two sockets was up to 6% slower than on one (hopper,
chess/susy): check `new19/new19-1s` in p7, since "both sockets must never be
slower than one" is part of the task.

The short-tree regression was traced (turing, `--profile`) to (a) warp-per-
segment kernels on root levels with ~49k-tile segments (`segmentPrefix`
0.27 -> 1.48 ms, `segmentBest` 0.15 -> 1.10 ms) and (b) lazy module loading of
a 16% larger fatbinary (+2.8 ms). Both fixed in `26e26335`.

### Turing (RTX 2080 Ti, one socket), new18 = `26e26335` before the merge

| case (train ms) | base | new15 | new18 |
|---|---|---|---|
| chess_100m cart d10 GPU-only one-pass | 344-356 | 362-367 | 322-338 |
| higgs c45 d10 hybrid one-pass | 656 | 682 | 636 |
| higgs cart no limit GPU-only (new15) | 6379 | 3779 | - |

`CUDA_MODULE_LOADING=LAZY` with new18 gives back the old setup time, i.e. ~24
ms of the gain is the eager loading. Golden tests pass on new19 (the merged
branch, `5dcaae59`).

## What is pending right now

**p7 A/B: SLURM jobs 667 (ampere) and 668 (hopper)**, submitted 2026-11-13
13:07, still PENDING at 18:00 behind user asawicki's jobs (635-640, running
since ~08:45). Exclusive node, `-t 8:00:00`. p6 took a few hours per node.

* variants: `~/numa-dt/ab/variants-p7-{ampere,hopper}.txt`:
  `main` = `~/numa-dt/main2/tree` (current main 55d83a48),
  `new19` = `~/numa-dt/new19/tree` (branch 5dcaae59), `main-1s`, `new19-1s`,
  `gpu-main`, `gpu-new19` (`--gpu-only`)
* cases: `~/numa-dt/ab/cases-p5.txt` (12: depth-10 CART/C4.5 on higgs, susy,
  mnist8m, sorel_4m, chess_100m; HIGGS CART/C4.5 without depth limit)
* output: `~/numa-dt/ab/p7-<node>/times.csv` and `out/`, logs
  `~/numa-dt/ab/p7-<node>.log`; submitted by `~/numa-dt/ab/p7.sh` (do not
  rerun it unless the jobs are gone: it submits new jobs)

If the jobs are gone from `squeue` and `times.csv` has 649 lines (header +
12 cases x 6 variants x 9), they finished.

## How to continue

1. Wait for 667/668 (`squeue -a -j 667,668`). Do not cancel other users' jobs.
2. Summarize each node:
   ```bash
   python3 ~/numa-dt/summ.py ~/numa-dt/ab/p7-ampere/times.csv main new19
   python3 ~/numa-dt/summ.py ~/numa-dt/ab/p7-hopper/times.csv main new19
   bash ~/numa-dt/peaks.sh ~/numa-dt/ab/p7-ampere/out   # peak RSS per variant-case
   ```
   Columns: medians per variant, then ratios new19/main, main-1s/main,
   new19-1s/main-1s, new19/new19-1s, new19/main-1s, gpu-new19/gpu-main, and the
   number of distinct tree shas per case (must be 1).
3. Acceptance: `shas` = 1 everywhere; `new19/main` <= ~+2% on every case
   (noise: compare with the run spread the script prints; chess/higgs depth-10
   cases are 60-210 ms and noisy on hopper, up to 30% spread); `new19/new19-1s`
   <= ~0 on every case (both sockets never slower than one); GPU-only HIGGS
   no-limit ~-55..-64%; peak memory not much above main.
4. If a case regresses: profile it with `--profile FILE` (one tree, no
   `--repeat`) for both binaries on that node (`srun -p <node> -w <node>
   --exclusive ...`), compare `phases`, `tables.gpu_levels` and the `gpu`
   kernel list (the earlier analysis scripts are gone; a few lines of Python
   over the JSON are enough). Build a fixed binary from the clone with
   `sbatch bench/cluster/build.sh` (or `srun -p turing -w turing -c 16 bash
   bench/cluster/build.sh`, `TREE_GIT_DESCRIBE=<name>` in the environment),
   copy `./tree` to `~/numa-dt/<newNN>/tree`, rerun `tests/check.sh`
   (`TREE_BIN=<binary> srun -p turing bash tests/check.sh`), then a new A/B
   with `ab.sh` (`sbatch -p <node> -w <node> --exclusive -t 8:00:00 -o <log>
   ab.sh <out> <variants> <cases> 3 3`).
5. Write the results into the branch:
   * `docs/PERFORMANCE.md`: a new top section "CUDA on two sockets and small
     nodes (2026-11)" with the p7 tables (both nodes, hybrid vs one socket vs
     GPU-only, peak memory), what changed, and the dropped experiments above.
     Keep the existing sections.
   * `docs/CUDA.md`: replace the placeholder line in "### Small nodes",
     `GPU-only HIGGS (10.5M × 28, no depth limit): A100 3.9 → see PERFORMANCE.md.`,
     with the measured numbers (A100 and H100, CART and C4.5).
   Commit them on `worktree-numa-gpu` with explicit paths.
6. **Present the results to the user** (tables per node, what changed, what
   was dropped) and ask for approval. Only on the user's yes:
   `git checkout main && git pull && git merge worktree-numa-gpu` (resolve
   `docs/PERFORMANCE.md` by keeping both top sections if main moved), rebuild
   and run `tests/check.sh` once more, `git push origin main`. Do not push the
   branch.
7. After the merge, tell the user: another session ("CART C45 benchmark suite
   for paper", on the user's PC) waits for numa-gpu on main to start the final
   benchmark rerun (`~/dt-final`, `final.sh`). That session cannot be messaged
   from mordor3, so the user has to pass it on.

## Files on mordor3 (`~/numa-dt`)

| path | what |
|---|---|
| `numa-gpu.bundle` | the branch's commits (see above) |
| `main2/tree` | current main 55d83a48, built 2026-11-13 (CUDA sm_75/80/90) |
| `new19/tree` | branch tip 5dcaae59 |
| `base/tree`, `new15/tree`, `new18/tree` | old main 1e655067; branch at 6c27d243; 26e26335 before the merge |
| `new/` | source tree of the branch as synced for builds (not a git checkout) |
| `mainsrc/` | source of main 55d83a48 (its `build.log` ends in an error about `bench/adapters/rusage.c`: only the rusage launcher failed, `./tree` was linked) |
| `based/` | old main with debug phases (ignore) |
| `build-new.sh <dir>` | builds `new/` on turing and copies `./tree` to `<dir>` |
| `build-main.sh` | builds `mainsrc/` to `main2/` |
| `ab/ab.sh` | interleaved A/B harness (also `bench/cluster/ab.sh` on the branch; usage in its header) |
| `ab/p6-*`, `ab/p7-*` | A/B outputs |
| `summ.py`, `peaks.sh` | summaries (above) |
| `short4.sh` | quick turing comparison with `--profile` (edit its variant list first) |

Datasets: `/home/bmircevski/dt/bench/data/<name>/train.csv` (higgs, susy,
mnist8m, sorel_4m, chess_100m, ...).

## Hardware notes

* ampere: 2x EPYC 7402 (24 cores each), 2x A100 80GB PCIe; the GPU used is on
  NUMA node 0.
* hopper: 2x Xeon 6444Y (16 cores each), H100; GPU on node 1.
* turing: one socket, RTX 2080 Ti; used for builds and quick checks (timings
  there do not predict A100/H100: pinned vs pageable went the other way).
* epyc/dell nodes have no GPU and no CUDA toolkit.
