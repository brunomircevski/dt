# Handoff: state on 2026-10-18 ~17:45 (machine shut down mid-task)

Everything up to the profiling work is done, committed and pushed to `main`. The one open task is the **fix of two slowdowns the profiling found**. It is unfinished: its code and data are on branch **`wip/arena-pagelock-fix`** (pushed), not on `main`.

## Done (on `main`, pushed)

| Commit | What |
|---|---|
| c94e3b7, f77d39d | `./tree --profile FILE` (src/core/trace.h): phases, per-thread work states with a 5 ms sampler, CUDA kernel/copy spans, per-level table |
| e345729 | `bench/profile_chart.py` and [bench/results/cluster-ecbdl-profile](bench/results/cluster-ecbdl-profile/BENCHMARK.md): ECBDL'14 15M profiled on hopper, ampere (CUDA) and epyc (parallel); `bench/cluster/profile.sh` |
| bcd22c0 | [cluster-tree-vs-gpu-libs](bench/results/cluster-tree-vs-gpu-libs/BENCHMARK.md): ECBDL'14 15M added (`chart-ecbdl_15m.png`); `gpubench.py` predicts cuML in 1M-row chunks (its predict overflows past 2^32 values) |
| e6a946b | `paper/` notes (from another session) |

**mordor3 is not fully in sync yet:** `~/dt` is at f77d39d. It has `bench/results/cluster-ecbdl-profile/` as untracked files (the same results that are now committed) and a modified `bench/cluster/build.log`. To sync it: `cd ~/dt && rm -r bench/results/cluster-ecbdl-profile && git checkout bench/cluster/build.log && git pull`, then rebuild (bench/cluster/README.md). `~/gpu-dt-bench` has the current scripts (same md5 as the repo).

## Open: the fix (branch `wip/arena-pagelock-fix`)

Two findings of [cluster-ecbdl-profile](bench/results/cluster-ecbdl-profile/BENCHMARK.md):

1. **Arena page-lock stall (CUDA)**, `src/build/gpu_grower.cu`. For unpruned CART and C4.5, setup()'s pinner thread page-locks the whole hand-off arena (75 GB on ECBDL 15M) with one `cudaHostRegister`. That holds the CUDA driver's lock and stalls the first GPU level or "grow setup".
   - **Change on the branch:** arenas up to `kPinAtOnceMaxBytes` (16 GB) are still registered at once. Larger ones go in 1 GB chunks, `kPinAheadBytes` (2 GB) ahead of the hand-offs, through `pinArena(touch)`.
2. **Count tables zero-filled on one thread**, `src/build/cpu_builder.{h,cpp}` and `cpu_grower.cpp`. `fillCountTables` did `vector::resize` on 2 × (rows + 1) doubles before its parallel fill (0.28 s on epyc for 15M rows).
   - **Change on the branch:** `CountTable = std::vector<double, UninitializedAllocator<double>>`.

### A/B results so far (`bench/results/wip-arena-pagelock/ab-*.log`)

`tree.base` is f77d39d; `tree` is the branch's code plus environment knobs for the variants (`gpu_grower-ab-knobs.diff`):
- `new`: the branch's defaults.
- `new-ah4`: `DT_PIN_AHEAD_GB=4`.
- `new-chunk`: `DT_PIN_AT_ONCE_MAX_GB=0`, i.e. airline also in chunks.

Train total in seconds (2–3 runs each). Every variant built the same trees.

| Case | hopper base → new | ampere base → new |
|---|---|---|
| ECBDL unpruned CART | 23.0 → 18.6 (−19%) | 25.7 → 24.2 (−6%) |
| ECBDL C4.5 | 51.7 → 47.5 (−8%) | 67.0 → 68.5 (+2%); `new-ah4` 67.2 |
| airline unpruned CART | 3.3–3.75 → 3.25; `new-chunk` 3.0–3.1 | 4.16 → 3.91; `new-chunk` 3.69 (1 run) |
| airline C4.5 | 5.05 → 5.2 (+3%) | (job was still running) |
| airline CART α | 1.11 → 1.15; `new-chunk` 1.10 | (job was still running) |

What the numbers say:
- **Level 0 is fixed:** 5.9 s → 0.15 s (hopper ECBDL unpruned) and 2.0 → 0.16 s (ampere).
- **But C4.5's GPU levels get 2.3–3.6 s slower with chunked page-locking** (hopper 25.4 → 27.8–28.2 s, ampere 33.9 → 36.4–37.5 s): each 1 GB registration still takes the driver lock while the levels run. That is the trade-off the old code comment describes (12 × 1 GB chunks made airline C4.5 0.4 s slower). On ampere it eats the whole gain; on hopper there is a net gain, because this time hopper's baseline stalled 5.7 s, not the 1.1 s of the profile run.
- **Unexplained:** airline C4.5 (12 GB arena, so still registered at once) moved its 0.2 s stall from grow setup into level 0, and its levels got 0.2 s slower in 2 of 3 runs.
- **Not measured yet:** the count-table change on epyc (CPU parallel), and whether it changes the trees there.

### Final verdict of the fix session (after the above)

- **Change 2 (count tables): keep.**
  - The golden tests pass, and the HIGGS trees (28 threads, CART α 1e-5 and C4.5) are byte-identical.
  - The single-thread stretch at the start of the build is gone. The train time on the desktop is unchanged within noise (HIGGS is small).
  - The CUDA grower only uses the small default tables. Not measured on epyc yet.
- **Change 1 (chunked page-locking): drop it as it is.**
  - It is faster on hopper and slower for C4.5 on ampere: registering chunks during the levels makes the level copies wait (`uploadLevel` 0.34 → 3.4 s). A 4 GB lead only brought C4.5 to 67.5 s.
  - It also has a bug. Forcing chunked mode on airline (`DT_PIN_AT_ONCE_MAX_GB=0`) crashed C4.5 after the grow with `CUDA error at gpu_grower.cu:1990: invalid argument` (line number in the knob build). Some later step assumes the arena is one registration, and ECBDL's chunked path may hit the same bug.
- **Ampere's registration benchmark** (arrived later; `~/dt-fix/reg-ampere.log`), 4 KB pages → `MADV_HUGEPAGE`:
  - register 75 GB: 5.4–5.8 s → 1.83 s;
  - touch 75 GB: 3.7 s → 0.66 s;
  - register 12 GB (airline-sized): 1.31 s → 0.31 s.

  Registering in 4 GB chunks costs nothing extra when it runs alone, on either node. So chunked mode's slowdown comes from registering while the GPU levels run.
- **Registration benchmark on hopper:** `cudaHostRegister` of a touched 75 GB buffer takes 2.67 s with 4 KB pages and 1.55 s with `MADV_HUGEPAGE`. Touching it drops from 2–3.5 s to 0.5 s. Hopper's transparent huge pages are set to `madvise`, so the arena gets no huge pages today. Log: `~/dt-fix/reg-hopper.log`.
- **Suggested approach:** `madvise(MADV_HUGEPAGE)` on the arena, keeping the single registration. Fix the chunked-mode crash before using any chunked mode. The alternative is staged hand-off copies, which need no arena registration.
- No `profile.sh` rerun was made and nothing of the fix is on `main`.

### Still running on mordor3 when this machine went down

- A/B jobs 932 (hopper) and 933 (ampere): `~/dt-fix/ab-<node>.log`, profiles in `~/dt-fix/ab/<node>/`.
- Queued: 934 and 935, `bench/cluster/regbench.sh` (`regbench.cu`). It times `cudaHostRegister` of 75 GB with 4 KB pages, with huge pages, and in 4 GB chunks, and of 12 GB. Output: `~/dt-fix/reg-<node>.log`. Its result was meant to decide the approach: if huge pages make one registration fast, register at once (with `madvise(MADV_HUGEPAGE)` on the arena) instead of in chunks.
- `~/dt-fix` is the fix session's own clone (not git-synced; copy files from it). `~/dt` was not touched.

### Next steps

1. Take change 2 (count tables) from the branch onto `main` (the three CPU files), and check it on epyc.
2. For change 1, try `MADV_HUGEPAGE` on the arena with the single registration. Measure on hopper and ampere: ECBDL unpruned CART and C4.5, airline, HIGGS. Huge pages should also shrink airline's 0.2–0.3 s registration. Drop the chunked mode, or fix its crash before keeping it.
3. Commit what is a net win to `main` (explicit paths: other sessions share the git index), push, then sync and rebuild `~/dt` on mordor3. Delete this file and the `wip/` branch when done.
