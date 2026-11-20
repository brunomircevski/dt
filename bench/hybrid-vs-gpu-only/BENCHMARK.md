# hybrid-vs-gpu-only: ./tree's CUDA hybrid against GPU only, by depth

![ampere](chart-ampere.png)

![hopper](chart-hopper.png)

**What.** ./tree `--hybrid` (a node with fewer rows than `--handoff-rows` is
copied to the host and its subtree grown by the CPU threads while the GPU goes
on with the next level) against `--gpu` (every node, down to the leaves, on
the GPU). CART and C4.5 without pruning, grown to depth 5, 10, 15, 20, 25, 30
and without a depth limit, 10 runs a case on hopper and 5 on ampere (the
chart's footnote gives the count), on three of the paper's datasets, chosen by a screen of five
([below](#earlier-measurements-95b6d77)): HIGGS (10.5M rows × 28 features),
where the hybrid's lead was the largest; SOREL-20M's 4M sample (4M × 2,381),
where it was the smallest; and MNIST8M (8.1M × 784) in between. The trees are
the same in both modes (node counts under each group). The hybrid runs at the
fastest `--handoff-rows` of a sweep on the same node and dataset
([the sweep](#--handoff-rows-the-sweep)). Its picks are now stored in
`bench/handoff-tuning/handoff_rows.json` ("nodepth"); later runs take them from there (and the
depth-10 sweep's value for depth 10) instead of sweeping again
([bench/tools/README.md](../tools/README.md#the-hybrids-hand-off-size)).

**Where.** The whole node (`--exclusive`), one GPU, every CPU of the node:
ampere, A100 80 GB PCIe (GPU 0, NUMA node 0), 48 threads (CPUs 0–23, then
24–47); hopper, H100 PCIe (GPU 2, NUMA node 1), 32 threads (16–31, then 0–15).
The GPU's socket's CPUs come first and the memory is not bound
([bench/tools/cluster/README.md](../tools/cluster/README.md#settings-specific-to-the-cluster)).

**When.** hopper: ./tree 81bf6e6c (the same tree code as f53347a9), 2026-11-19,
10 runs a case, in two jobs (`hopper-higgs-sorel_4m.log`, `hopper-mnist8m.log`;
`hybrid_vs_gpu_only.sh` with `RUNS=10`), taking the `--handoff-rows` of the
2026-11-15/16 sweep from `bench/handoff-tuning/handoff_rows.json` instead of
sweeping again. The CUDA runs time ./tree's GPU setup (memory pool, staging
buffers, and for `--hybrid` its page-locked hand-off buffers): only the bare
CUDA context is untimed ([bench/tools/README.md](../tools/README.md#how-time-is-measured)).
ampere: ./tree f53347a9 (`src/` = d2957ff2), one build, 2026-11-15/16, by
`bench/tools/cluster/final.sh -m 5` (`hybrid_vs_gpu_only.sh` after the GPU
node's test-set runs; `ampere.log`), with the GPU setup untimed: **out of
date** (⚠, faded on the chart), not rerun: ampere was unavailable from
2026-11-19 for two weeks. The sweep (both nodes) is from 2026-11-15/16; SOREL
4M's was cut short to save time: ampere skipped 1024 and 2048, hopper 2048
(the curve falls steadily from 512 to 16384, so they could not have won); the
job was resubmitted with the remaining values and kept the ones already
measured. hopper's 2026-11-15/16 runs (f53347a9, 5 runs a case) are in git at
84b4c67e; the runs before (95b6d77, 3 runs a case, 2026-11-06/08) at 1e655067.

## Results

Mean training time of the runs (GPU setup, presort, build; not reading the
data): 10 on hopper, 5 on ampere (out of date, GPU setup untimed). Hybrid / GPU
only, GPU only / hybrid in brackets (above 1: the hybrid is faster).

### ampere (out of date: f53347a9, GPU setup untimed)

| Depth | HIGGS CART | HIGGS C4.5 | MNIST8M CART | MNIST8M C4.5 | SOREL 4M CART | SOREL 4M C4.5 |
|---|---|---|---|---|---|---|
| 5 | 0.123 / 0.120 s (0.97×) | 0.135 / 0.130 s (0.96×) | 2.06 / 2.03 s (0.99×) | 2.07 / 2.04 s (0.99×) | 2.99 / 2.89 s (0.97×) | 3.27 / 3.25 s (0.99×) |
| 10 | 0.176 / 0.167 s (0.95×) | 0.196 / 0.190 s (0.97×) | 3.27 / 3.11 s (0.95×) | 3.07 / 3.12 s (1.01×) | 4.42 / 4.18 s (0.94×) | 4.50 / 4.59 s (1.02×) |
| 15 | 0.262 / 0.273 s (1.04×) | 0.272 / 0.285 s (1.05×) | 4.86 / 4.97 s (1.02×) | 4.33 / 4.47 s (1.03×) | 5.95 / 5.81 s (0.98×) | 5.74 / 5.81 s (1.01×) |
| 20 | 0.438 / 0.529 s (1.21×) | 0.374 / 0.456 s (1.22×) | 5.87 / 7.26 s (1.24×) | 5.41 / 6.25 s (1.16×) | 7.27 / 7.47 s (1.03×) | 6.91 / 7.11 s (1.03×) |
| 25 | 0.527 / 0.900 s (1.71×) | 0.462 / 0.702 s (1.52×) | 6.93 / 9.01 s (1.30×) | 6.39 / 8.18 s (1.28×) | 9.09 / 9.06 s (1.00×) | 8.13 / 8.42 s (1.03×) |
| 30 | 0.646 / 1.21 s (1.88×) | 0.507 / 0.933 s (1.84×) | 7.69 / 10.0 s (1.31×) | 7.13 / 9.78 s (1.37×) | 11.2 / 10.6 s (0.94×) | 9.32 / 9.77 s (1.05×) |
| none | 0.770 / 1.59 s (2.06×) | 0.651 / 1.51 s (2.32×) | 10.4 / 14.8 s (1.42×) | 12.2 / 18.0 s (1.47×) | 28.7 / 33.3 s (1.16×) | 73.5 / 79.4 s (1.08×) |

### hopper

| Depth | HIGGS CART | HIGGS C4.5 | MNIST8M CART | MNIST8M C4.5 | SOREL 4M CART | SOREL 4M C4.5 |
|---|---|---|---|---|---|---|
| 5 | 0.299 / 0.180 s (0.60×) | 0.309 / 0.186 s (0.60×) | 1.55 / 1.43 s (0.92×) | 1.59 / 1.47 s (0.93×) | 2.16 / 2.04 s (0.94×) | 2.48 / 2.36 s (0.95×) |
| 10 | 0.328 / 0.209 s (0.64×) | 0.353 / 0.239 s (0.68×) | 2.38 / 2.26 s (0.95×) | 2.38 / 2.30 s (0.97×) | 3.06 / 2.94 s (0.96×) | 3.37 / 3.26 s (0.97×) |
| 15 | 0.403 / 0.297 s (0.74×) | 0.415 / 0.301 s (0.73×) | 3.71 / 3.98 s (1.07×) | 3.28 / 3.41 s (1.04×) | 4.12 / 4.13 s (1.00×) | 4.21 / 4.20 s (1.00×) |
| 20 | 0.575 / 0.508 s (0.88×) | 0.500 / 0.463 s (0.93×) | 4.94 / 6.09 s (1.23×) | 4.11 / 5.04 s (1.23×) | 5.07 / 5.46 s (1.08×) | 5.07 / 5.12 s (1.01×) |
| 25 | 0.731 / 0.781 s (1.07×) | 0.595 / 0.669 s (1.12×) | 5.84 / 7.58 s (1.30×) | 4.94 / 6.91 s (1.40×) | 5.82 / 6.64 s (1.14×) | 5.90 / 6.06 s (1.03×) |
| 30 | 0.854 / 0.951 s (1.11×) | 0.670 / 0.877 s (1.31×) | 6.45 / 8.50 s (1.32×) | 5.55 / 8.66 s (1.56×) | 6.54 / 7.75 s (1.19×) | 6.75 / 6.98 s (1.03×) |
| none | 1.06 / 1.26 s (1.19×) | 0.837 / 1.48 s (1.77×) | 8.04 / 12.3 s (1.53×) | 9.58 / 15.3 s (1.60×) | 20.0 / 23.9 s (1.20×) | 52.4 / 56.6 s (1.08×) |

- **On hopper GPU only now wins the shallow trees.** Both modes time their GPU
  setup now, and the hybrid's setup also makes its page-locked hand-off
  buffers: 0.12 s more than GPU only on every dataset (HIGGS 0.27 against
  0.15 s, MNIST8M 0.95 against 0.83 s, SOREL 4M 1.40 against 1.28 s). Up to
  depth 10 the builds are the same, so GPU only is faster: on HIGGS, whose
  trees take 0.2–0.4 s, by a lot (0.60–0.68×), on MNIST8M and SOREL 4M by
  3–8% (0.92–0.97×). HIGGS's hybrid stays behind to depth 20 (0.88–0.93×) and
  leads from depth 25 (1.07–1.12×); MNIST8M and SOREL 4M break even at depth 15.
  Hopper's runs spread at most 6% (HIGGS up to 21%, in runs of 0.2–1 s).
- **Deeper, the hybrid leads:** without a depth limit on hopper HIGGS 1.19×
  (CART) and 1.77× (C4.5), MNIST8M 1.53× and 1.60×, SOREL 4M 1.20× and 1.08×.
  On ampere (out of date, setup untimed) HIGGS 2.1–2.3×, MNIST8M 1.4–1.5×,
  SOREL 4M 1.08–1.16×; before d2957ff2 (95b6d77) HIGGS was 4.3–4.9×.
- **Because GPU only got faster in the deep trees**, not the hybrid: GPU only
  at no limit, HIGGS CART 3.97 → 1.59 s on ampere and 4.01 → 1.26 s on hopper
  (95b6d77 → now), MNIST8M CART 29.6 → 14.8 s and 24.5 → 12.3 s; the hybrid
  HIGGS CART 0.82 → 0.77 s on ampere and 0.81 → 1.06 s on hopper (now with its
  0.27 s of setup). The deep levels of a few small nodes were GPU only's
  weakness (a whole level's kernels and synchronisation for a few rows);
  d2957ff2 skips the tiles without a cut and the earlier small-node kernels
  (bc9ea314) shorten those levels. HIGGS's unlimited trees have 2.9M (CART)
  and 1.3M (C4.5) nodes over 69–71 levels; the hybrid's CPU threads still grow
  those subtrees in one go.
- **SOREL 4M barely gains (0.94–1.20×).** Its trees are small (88k CART
  nodes, 70k C4.5) and long and thin: no limit means depth 231 (CART) and
  572 (C4.5), the levels past 30 peeling a few rows a time off nodes that
  stay above `--handoff-rows`. Those nodes, and every level's 2,381
  features, stay on the GPU in both modes. On ampere at depth 25–30 CART the
  hybrid was no faster (1.00× and 0.94×).
- **Past the setup, the gap is all in the build.** Build at no depth limit on
  hopper: HIGGS CART 0.79 s hybrid, 1.11 s GPU only (C4.5 0.55 and 1.30 s);
  MNIST8M CART 7.1 s, 11.5 s; SOREL 4M CART 18.6 s, 22.6 s. On ampere: HIGGS
  CART 0.69 s, 1.51 s; MNIST8M CART 9.2 s, 13.6 s; SOREL 4M CART 26.9 s, 31.5 s.
- **hopper against ampere:** where the GPU does the work the H100 is
  1.4–1.5× faster even with its setup timed (MNIST8M depth 10: 2.26–2.38
  against 3.11–3.27 s; SOREL 4M at no limit 20.0 / 52.4 s against 28.7 /
  73.5 s). HIGGS's hybrid at no limit is CPU bound: 0.79 s of build on hopper
  with 32 threads, 0.69 s on ampere with 48.
- **VRAM:** the hybrid needs a little less (HIGGS 3.4 GiB against 3.7–4.3,
  MNIST8M 48.9 against 50.1–51.2, SOREL 4M 75.9 GiB against 76.4 of the 80 GB).

## Who is working (MNIST8M, profiled)

![profile, ampere](chart-profile-ampere-mnist8m.png)

![profile, hopper](chart-profile-hopper-mnist8m.png)

The MNIST8M depth runs once more with `./tree --profile`, 10 runs a case on
hopper (5 on ampere, out of date), in the same job as the plain runs
(`profile/{cuda,gpuonly}-<node>-mnist8m/`). Profiling costs little: every
profiled mean is within 3% of the plain run's on both nodes.

Each millisecond of training goes to one segment
([hybrid_profile_chart.py](../tools/hybrid_profile_chart.py)):
- **Setup:** before growing: GPU context and memory, upload, presort.
- **GPU alone:** a GPU kernel runs and no CPU thread grows a subtree.
- **GPU + CPU at once:** a kernel runs while CPU threads grow subtrees (hybrid only).
- **CPU alone:** no kernel or copy runs, CPU threads grow subtrees.
- **Transfer:** a host↔GPU copy runs with no kernel, or the host stages copies.
- **Host:** none of these: launching kernels, waiting, assembling the tree, C4.5's tidy-up.

CPU subtree work is the threads sampled (every 5 ms) in sweep, partition,
mark, build or leaves; under each bar its thread-seconds.

- **Setup:** 1.2 s on ampere (setup untimed there: the same in both modes);
  on hopper 0.98–1.02 s for the hybrid and 0.85–0.89 s for GPU only, the
  hand-off buffers. 58–62% of a depth-5 tree, 6–12% at no limit.
- **The CPU's subtree work grows with depth.** On ampere CART 0 → 5 → 69 →
  139 → 188 → 222 → 299 thread-seconds (depth 5 → no limit), C4.5 0 → 2 → 21 →
  65 → 120 → 167 → 296; on hopper CART 177 and C4.5 186 at no limit. At no limit that is
  27–32 of ampere's 48 threads and 22–25 of hopper's 32 busy on average while
  growing.
- **Most of it runs alongside the GPU:** the GPU + CPU segment grows to 60%
  (CART) and 72% (C4.5) of the hybrid's time on ampere at no limit, 62% and
  74% on hopper. **CPU alone**, the GPU waiting for the threads, is larger
  than before: on ampere up to 1.10 s (14%, CART depth 30), on hopper 1.14–1.75 s
  (23–27%) for CART at depth 20–30. With the faster GPU levels the GPU now
  waits for the CPU threads, most on hopper with its 32.
- **GPU only does that work on the GPU instead:** its GPU segment at no
  limit is 11.6 s (CART) and 14.0 s (C4.5) on ampere, against the hybrid's
  7.7 s and 9.8 s of GPU time (alone + with CPU). Its host time (launching and
  synchronising many short levels) is 1.2–2.0 s (8–11%) at no limit on
  ampere and 1.7–2.7 s (13–18%) on hopper, where the hybrid's is under 0.1 s.
- **The hybrid pays 0.3–1.2 s of transfer** on ampere from depth 15 on (the
  small nodes' columns copied to the host), 0.2–0.4 s on hopper, against
  0.05–0.8 s for GPU only.
- Up to depth 10 the two modes' bars are the same, but for the hybrid's longer setup on hopper.

The hybrid alone on ampere, in seconds on a linear axis shared by both panels,
and as shares of each bar:

![hybrid profile, seconds](chart-profile-hybrid-seconds-ampere-mnist8m.png)

![hybrid profile, percent](chart-profile-hybrid-percent-ampere-mnist8m.png)

Table, ampere (out of date, GPU setup untimed; seconds, share of the training time in brackets;
`hybrid_profile_chart.py` prints it):

| Algorithm | Depth | Mode | runs | train s | without --profile s | nodes | Setup (GPU init, upload, presort) s (%) | GPU working alone s (%) | GPU + CPU working at once s (%) | CPU working alone s (%) | CPU↔GPU transfer s (%) | Host: launching, waiting, finishing s (%) | CPU subtree thr·s |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CART | 5 | hybrid | 5 | 2.09 | 2.06 | 63 | 1.24 (59) | 0.81 (39) | 0.00 (0) | 0.00 (0) | 0.03 (1) | 0.02 (1) | 0 |
| CART | 5 | GPU only | 5 | 2.05 | 2.03 | 63 | 1.22 (59) | 0.81 (39) | 0.00 (0) | 0.00 (0) | 0.00 (0) | 0.03 (1) | 0 |
| CART | 10 | hybrid | 5 | 3.27 | 3.27 | 2,047 | 1.18 (36) | 1.62 (49) | 0.17 (5) | 0.02 (1) | 0.25 (8) | 0.03 (1) | 5 |
| CART | 10 | GPU only | 5 | 3.07 | 3.11 | 2,047 | 1.18 (38) | 1.83 (60) | 0.00 (0) | 0.00 (0) | 0.01 (0) | 0.05 (2) | 0 |
| CART | 15 | hybrid | 5 | 4.87 | 4.86 | 44,359 | 1.22 (25) | 1.56 (32) | 1.00 (21) | 0.11 (2) | 0.95 (19) | 0.03 (1) | 69 |
| CART | 15 | GPU only | 5 | 5.01 | 4.97 | 44,359 | 1.22 (24) | 3.37 (67) | 0.00 (0) | 0.00 (0) | 0.13 (3) | 0.29 (6) | 0 |
| CART | 20 | hybrid | 5 | 5.95 | 5.87 | 168,241 | 1.22 (20) | 1.37 (23) | 1.81 (30) | 0.45 (8) | 1.07 (18) | 0.03 (1) | 139 |
| CART | 20 | GPU only | 5 | 7.35 | 7.26 | 168,241 | 1.23 (17) | 5.21 (71) | 0.00 (0) | 0.00 (0) | 0.41 (6) | 0.50 (7) | 0 |
| CART | 25 | hybrid | 5 | 6.98 | 6.93 | 231,991 | 1.22 (17) | 1.37 (20) | 2.38 (34) | 0.89 (13) | 1.08 (15) | 0.03 (0) | 188 |
| CART | 25 | GPU only | 5 | 8.89 | 9.01 | 231,991 | 1.19 (13) | 6.51 (73) | 0.00 (0) | 0.00 (0) | 0.50 (6) | 0.70 (8) | 0 |
| CART | 30 | hybrid | 5 | 7.70 | 7.69 | 257,021 | 1.19 (15) | 1.37 (18) | 2.92 (38) | 1.10 (14) | 1.09 (14) | 0.04 (0) | 222 |
| CART | 30 | GPU only | 5 | 10.08 | 10.03 | 257,021 | 1.21 (12) | 7.46 (74) | 0.00 (0) | 0.00 (0) | 0.55 (6) | 0.86 (9) | 0 |
| CART | none | hybrid | 5 | 10.42 | 10.39 | 301,939 | 1.21 (12) | 1.39 (13) | 6.29 (60) | 0.31 (3) | 1.17 (11) | 0.05 (0) | 299 |
| CART | none | GPU only | 5 | 14.69 | 14.78 | 301,939 | 1.18 (8) | 11.64 (79) | 0.00 (0) | 0.00 (0) | 0.67 (5) | 1.20 (8) | 0 |
| C4.5 | 5 | hybrid | 5 | 2.06 | 2.07 | 61 | 1.18 (57) | 0.81 (39) | 0.01 (1) | 0.00 (0) | 0.03 (1) | 0.03 (1) | 0 |
| C4.5 | 5 | GPU only | 5 | 2.05 | 2.04 | 61 | 1.20 (58) | 0.82 (40) | 0.00 (0) | 0.00 (0) | 0.00 (0) | 0.03 (2) | 0 |
| C4.5 | 10 | hybrid | 5 | 3.15 | 3.07 | 1,453 | 1.25 (40) | 1.43 (45) | 0.38 (12) | 0.01 (0) | 0.05 (2) | 0.03 (1) | 2 |
| C4.5 | 10 | GPU only | 5 | 3.12 | 3.12 | 1,453 | 1.21 (39) | 1.84 (59) | 0.00 (0) | 0.00 (0) | 0.01 (0) | 0.06 (2) | 0 |
| C4.5 | 15 | hybrid | 5 | 4.36 | 4.33 | 12,313 | 1.21 (28) | 1.36 (31) | 1.41 (32) | 0.03 (1) | 0.32 (7) | 0.04 (1) | 21 |
| C4.5 | 15 | GPU only | 5 | 4.47 | 4.47 | 12,313 | 1.20 (27) | 3.04 (68) | 0.00 (0) | 0.00 (0) | 0.05 (1) | 0.18 (4) | 0 |
| C4.5 | 20 | hybrid | 5 | 5.44 | 5.41 | 42,915 | 1.23 (23) | 1.07 (20) | 2.51 (46) | 0.06 (1) | 0.54 (10) | 0.04 (1) | 65 |
| C4.5 | 20 | GPU only | 5 | 6.30 | 6.25 | 42,915 | 1.21 (19) | 4.47 (71) | 0.00 (0) | 0.00 (0) | 0.17 (3) | 0.46 (7) | 0 |
| C4.5 | 25 | hybrid | 5 | 6.39 | 6.39 | 84,917 | 1.22 (19) | 1.02 (16) | 3.26 (51) | 0.16 (3) | 0.70 (11) | 0.04 (1) | 120 |
| C4.5 | 25 | GPU only | 5 | 8.14 | 8.18 | 84,917 | 1.21 (15) | 5.93 (73) | 0.00 (0) | 0.00 (0) | 0.29 (4) | 0.71 (9) | 0 |
| C4.5 | 30 | hybrid | 5 | 7.11 | 7.13 | 123,289 | 1.19 (17) | 0.96 (14) | 3.94 (55) | 0.21 (3) | 0.77 (11) | 0.04 (1) | 167 |
| C4.5 | 30 | GPU only | 5 | 9.81 | 9.78 | 123,289 | 1.21 (12) | 7.24 (74) | 0.00 (0) | 0.00 (0) | 0.39 (4) | 0.97 (10) | 0 |
| C4.5 | none | hybrid | 5 | 12.29 | 12.22 | 218,325 | 1.23 (10) | 0.96 (8) | 8.83 (72) | 0.31 (2) | 0.89 (7) | 0.07 (1) | 296 |
| C4.5 | none | GPU only | 5 | 17.97 | 17.99 | 218,325 | 1.22 (7) | 13.97 (78) | 0.00 (0) | 0.00 (0) | 0.76 (4) | 2.02 (11) | 0 |

Table, hopper:

| Algorithm | Depth | Mode | runs | train s | without --profile s | nodes | Setup (GPU init, upload, presort) s (%) | GPU working alone s (%) | GPU + CPU working at once s (%) | CPU working alone s (%) | CPU↔GPU transfer s (%) | Host: launching, waiting, finishing s (%) | CPU subtree thr·s |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CART | 5 | hybrid | 10 | 1.59 | 1.55 | 63 | 0.98 (62) | 0.58 (37) | 0.00 (0) | 0.00 (0) | 0.01 (1) | 0.02 (1) | 0 |
| CART | 5 | GPU only | 10 | 1.45 | 1.43 | 63 | 0.85 (58) | 0.58 (40) | 0.00 (0) | 0.00 (0) | 0.00 (0) | 0.02 (2) | 0 |
| CART | 10 | hybrid | 10 | 2.42 | 2.38 | 2,047 | 0.99 (41) | 1.33 (55) | 0.05 (2) | 0.00 (0) | 0.03 (1) | 0.04 (1) | 1 |
| CART | 10 | GPU only | 10 | 2.30 | 2.26 | 2,047 | 0.86 (37) | 1.39 (60) | 0.00 (0) | 0.00 (0) | 0.01 (0) | 0.05 (2) | 0 |
| CART | 15 | hybrid | 10 | 3.77 | 3.71 | 44,359 | 0.99 (26) | 1.08 (29) | 0.88 (23) | 0.42 (11) | 0.37 (10) | 0.02 (1) | 46 |
| CART | 15 | GPU only | 10 | 3.99 | 3.98 | 44,359 | 0.85 (21) | 2.68 (67) | 0.00 (0) | 0.00 (0) | 0.07 (2) | 0.39 (10) | 0 |
| CART | 20 | hybrid | 10 | 4.99 | 4.94 | 168,241 | 0.99 (20) | 1.00 (20) | 1.45 (29) | 1.14 (23) | 0.38 (8) | 0.03 (1) | 89 |
| CART | 20 | GPU only | 10 | 6.15 | 6.09 | 168,241 | 0.86 (14) | 4.20 (68) | 0.00 (0) | 0.00 (0) | 0.22 (4) | 0.87 (14) | 0 |
| CART | 25 | hybrid | 10 | 5.90 | 5.84 | 231,991 | 0.98 (17) | 0.99 (17) | 1.93 (33) | 1.58 (27) | 0.39 (7) | 0.03 (0) | 117 |
| CART | 25 | GPU only | 10 | 7.64 | 7.58 | 231,991 | 0.86 (11) | 5.29 (69) | 0.00 (0) | 0.00 (0) | 0.26 (3) | 1.23 (16) | 0 |
| CART | 30 | hybrid | 10 | 6.49 | 6.45 | 257,021 | 0.98 (15) | 0.99 (15) | 2.37 (36) | 1.75 (27) | 0.38 (6) | 0.03 (0) | 136 |
| CART | 30 | GPU only | 10 | 8.59 | 8.50 | 257,021 | 0.85 (10) | 6.09 (71) | 0.00 (0) | 0.00 (0) | 0.29 (3) | 1.36 (16) | 0 |
| CART | none | hybrid | 10 | 8.09 | 8.04 | 301,939 | 0.99 (12) | 1.00 (12) | 5.03 (62) | 0.63 (8) | 0.41 (5) | 0.04 (0) | 177 |
| CART | none | GPU only | 10 | 12.30 | 12.30 | 301,939 | 0.85 (7) | 9.47 (77) | 0.00 (0) | 0.00 (0) | 0.34 (3) | 1.65 (13) | 0 |
| C4.5 | 5 | hybrid | 10 | 1.64 | 1.59 | 61 | 1.01 (62) | 0.60 (36) | 0.00 (0) | 0.00 (0) | 0.01 (1) | 0.02 (1) | 0 |
| C4.5 | 5 | GPU only | 10 | 1.51 | 1.47 | 61 | 0.89 (58) | 0.60 (39) | 0.00 (0) | 0.00 (0) | 0.00 (0) | 0.03 (2) | 0 |
| C4.5 | 10 | hybrid | 10 | 2.43 | 2.38 | 1,453 | 1.01 (41) | 1.28 (53) | 0.10 (4) | 0.00 (0) | 0.02 (1) | 0.03 (1) | 1 |
| C4.5 | 10 | GPU only | 10 | 2.31 | 2.30 | 1,453 | 0.86 (37) | 1.39 (60) | 0.00 (0) | 0.00 (0) | 0.01 (0) | 0.05 (2) | 0 |
| C4.5 | 15 | hybrid | 10 | 3.35 | 3.28 | 12,313 | 1.01 (30) | 1.12 (33) | 1.00 (30) | 0.02 (1) | 0.16 (5) | 0.03 (1) | 15 |
| C4.5 | 15 | GPU only | 10 | 3.45 | 3.41 | 12,313 | 0.88 (26) | 2.37 (69) | 0.00 (0) | 0.00 (0) | 0.03 (1) | 0.17 (5) | 0 |
| C4.5 | 20 | hybrid | 10 | 4.17 | 4.11 | 42,915 | 1.01 (24) | 0.83 (20) | 1.95 (47) | 0.12 (3) | 0.24 (6) | 0.03 (1) | 43 |
| C4.5 | 20 | GPU only | 10 | 5.10 | 5.04 | 42,915 | 0.89 (17) | 3.56 (70) | 0.00 (0) | 0.00 (0) | 0.09 (2) | 0.56 (11) | 0 |
| C4.5 | 25 | hybrid | 10 | 5.02 | 4.94 | 84,917 | 1.01 (20) | 0.77 (15) | 2.57 (51) | 0.37 (7) | 0.26 (5) | 0.04 (1) | 76 |
| C4.5 | 25 | GPU only | 10 | 6.96 | 6.91 | 84,917 | 0.88 (13) | 4.80 (69) | 0.00 (0) | 0.00 (0) | 0.16 (2) | 1.13 (16) | 0 |
| C4.5 | 30 | hybrid | 10 | 5.60 | 5.55 | 123,289 | 1.01 (18) | 0.58 (10) | 3.27 (58) | 0.43 (8) | 0.28 (5) | 0.03 (1) | 104 |
| C4.5 | 30 | GPU only | 10 | 8.72 | 8.66 | 123,289 | 0.89 (10) | 5.91 (68) | 0.00 (0) | 0.00 (0) | 0.25 (3) | 1.68 (19) | 0 |
| C4.5 | none | hybrid | 10 | 9.67 | 9.58 | 218,325 | 1.02 (11) | 0.65 (7) | 7.13 (74) | 0.51 (5) | 0.31 (3) | 0.06 (1) | 186 |
| C4.5 | none | GPU only | 10 | 15.46 | 15.33 | 218,325 | 0.88 (6) | 11.44 (74) | 0.00 (0) | 0.00 (0) | 0.41 (3) | 2.72 (18) | 0 |

## --handoff-rows: the sweep

Before the depth runs, each job runs the hybrid at no depth limit, CART and
C4.5, at ./tree's default (CART rows/8192 clamped to 1,024–2,048, C4.5
rows/4096 clamped to 1,024–4,096) and at fixed values, 5 runs a value, not
interleaved (`hybrid_vs_gpu_only.sh`). The value taken has the smallest mean,
over CART and C4.5, of its time over that algorithm's fastest. Results:
`sweep/cuda-<node>-<dataset>-mr<value>/`, summary `sweep/<node>-<dataset>.json`.

ampere (48 threads):

| HIGGS | default | 256 | **512** | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 0.785 | 0.848 | **0.767** | 0.805 | 0.815 | 0.906 | 1.06 | 1.25 | 1.44 |
| C4.5 (s) | 0.719 | 0.749 | **0.659** | 0.657 | 0.656 | 0.714 | 0.822 | 0.986 | 1.19 |

| MNIST8M | default | 256 | 512 | 1024 | 2048 | 4096 | **8192** | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 11.9 | 13.3 | 12.6 | 11.9 | 11.4 | 10.9 | **10.4** | 13.6 | 19.4 |
| C4.5 (s) | 13.3 | 16.1 | 15.0 | 14.0 | 13.3 | 12.6 | **12.2** | 14.6 | 20.1 |

| SOREL 4M | default | 256 | 512 | 4096 | 8192 | **16384** | 32768 |
|---|---|---|---|---|---|---|---|
| CART (s) | 31.2 | 32.3 | 31.7 | 29.5 | 29.2 | **28.7** | 46.7 |
| C4.5 (s) | 76.2 | 77.0 | 76.5 | 74.9 | 74.3 | **73.6** | 73.1 |

hopper (32 threads):

| HIGGS | default | **256** | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 0.813 | **0.790** | 0.794 | 0.808 | 0.822 | 0.901 | 1.02 | 1.17 | 1.29 |
| C4.5 (s) | 0.682 | **0.619** | 0.640 | 0.643 | 0.671 | 0.714 | 0.795 | 0.925 | 1.06 |

| MNIST8M | default | 256 | 512 | 1024 | 2048 | 4096 | **8192** | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 8.96 | 10.1 | 9.51 | 9.10 | 8.67 | 8.21 | **7.76** | 9.87 | 14.1 |
| C4.5 (s) | 10.2 | 12.4 | 11.5 | 10.9 | 10.2 | 9.79 | **9.35** | 9.79 | 13.1 |

| SOREL 4M | default | 256 | 512 | 1024 | 4096 | 8192 | **16384** | 32768 |
|---|---|---|---|---|---|---|---|---|
| CART (s) | 21.6 | 22.3 | 21.9 | 21.6 | 20.5 | 20.2 | **19.6** | 26.2 |
| C4.5 (s) | 54.0 | 54.7 | 54.3 | 53.9 | 53.2 | 52.6 | **52.1** | 52.6 |

- **HIGGS takes 512 on ampere and 256 on hopper**, and its minimum is flat:
  512–2,048 are within 6% of each other on ampere, 256–1,024 within 4% on
  hopper; ./tree's default (1,281 / 2,563) is 2–10% slower than the best. Before, both nodes took 512.
- **MNIST8M takes 8,192 on both nodes** (16,384 before, with the older GPU
  code): 16,384 is now 27–31% slower for CART. 8,192 is 13–15% (CART) and
  9% (C4.5) faster than the default (1,024 / 1,977).
- **SOREL 4M takes 16,384 on both nodes, as before.** It moves little (CART
  9–10%, C4.5 4% against the default, 1,024), as most of its work never
  reaches `--handoff-rows`; C4.5 alone would take 32,768 on ampere (under 1%
  faster), where CART is 34–63% slower.
- **The default here is 6–12% slower than the best** on every dataset. So
  after this run ./tree's default became 10 × features, between 512 and
  16,384 (HIGGS 512, MNIST8M 7,840, SOREL 4M 16,384): within 2% of the best on
  all three datasets and both nodes, judging by the sweep above (not rerun).
  The "default" columns and the depth runs here are the old rows-based default
  or the sweep's pick, so none of these numbers change.

## Earlier measurements (95b6d77)

The sections below were measured on 95b6d77 (2026-11-03/08, before d2957ff2
and the small-node kernels) with `--gpu-min-rows`, the flag's old name, and
were not rerun. The folders `sweep-one-socket/` and `screen-both-sockets/` hold
their runs, and `*-both-sockets.log`, `*-higgs-sorel.log` and
`hopper-one-socket.log` are those jobs' logs.

### Both sockets against one

Until 2026-11-06 every CUDA run of the cluster had only the CPUs and memory
of the GPU's socket (ampere 24 threads, hopper 16). MNIST8M's depth 15 and
deeper were then run again on both sockets; the hybrid's time, one socket /
both (above 1: both faster), with each placement's own swept
`--gpu-min-rows`:

| Depth | ampere MNIST8M CART | C4.5 | hopper MNIST8M CART | C4.5 |
|---|--:|--:|--:|--:|
| 15 | 1.05 | 1.03 | 1.04 | 0.95 |
| 20 | 1.13 | 1.06 | 1.18 | 1.00 |
| 25 | 1.18 | 1.08 | 1.22 | 1.04 |
| 30 | 1.14 | 1.07 | 1.21 | 1.04 |
| none | 1.11 | 1.06 | 1.12 | 1.04 |

- **The second socket pays off where the CPU grows many subtrees:** MNIST8M
  CART from depth 20 (1.11–1.22×); its C4.5 gains 0–8%. airline, in the
  benchmark until 9af6148, gained 1.15–1.37× from depth 25.
- **GPU only barely changes** (ampere 0.97–1.03×, hopper 0.94–1.02×), so the
  hybrid's lead over it grows.
- **The sweep moves up:** with more threads the CPU takes bigger nodes well,
  so MNIST8M's best `--gpu-min-rows` doubles (8,192 → 16,384 on both nodes).
- A test before these runs (ampere, HIGGS, SUSY, airline and MNIST8M, 3 runs,
  not kept) found both sockets 1.2–1.5× faster at no depth limit on HIGGS,
  SUSY and airline CART, and tried both sockets' CPUs with the memory bound
  to the GPU's node: slower than not binding it in every deep case.

### The one-socket sweeps of MNIST8M

The first, one-socket sweeps of MNIST8M (the value of its depth 5 and 10
runs) are in `sweep-one-socket/`: ampere by hand through `tune.sh`,
interleaved (`numactl --membind=0 taskset -c 0-23`, `--threads 24`;
`ampere-mnist8m.csv`, `ampere-mnist8m.json`), hopper by
`hybrid_vs_gpu_only.sh` (job 60, `cuda-hopper-mnist8m-*` and
`hopper-mnist8m.json`). Both took 8,192, half of the both-socket value: with
twice the threads the CPU keeps up with bigger nodes.

| one socket | default | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| ampere MNIST8M CART (s) | 21.4 | – | 22.3 | 21.2 | 20.5 | 20.0 | **19.0** | 20.5 | 27.0 |
| ampere MNIST8M C4.5 (s) | 25.3 | – | 27.7 | 26.3 | 25.5 | 24.2 | **23.4** | 22.7 | 26.9 |
| hopper MNIST8M CART (s) | 16.6 | 18.7 | 17.4 | 16.6 | 16.0 | 15.5 | **14.5** | 17.0 | 23.7 |
| hopper MNIST8M C4.5 (s) | 19.6 | 23.5 | 22.1 | 20.6 | 19.6 | 18.9 | **17.9** | 17.7 | 23.2 |

### Datasets: the screen

A screen on ampere, both sockets, no depth limit, unpruned, ./tree's default
`--gpu-min-rows`, one run each (job 214, 2026-11-07,
`screen-both-sockets/`), over the paper's datasets that fit in the A100's
80 GB. GPU-only time / hybrid time:

| Dataset | CART hybrid / GPU only | ratio | C4.5 hybrid / GPU only | ratio | nodes (CART / C4.5) |
|---|---|--:|---|--:|---|
| **HIGGS** | 0.86 / 4.15 s | **4.8×** | 0.67 / 2.99 s | **4.5×** | 2.86M / 1.32M |
| SUSY | 0.35 / 1.26 s | 3.6× | 0.17 / 0.20 s | 1.2× | 1.19M / 25k |
| ECBDL'14 15M | 19.8 / 30.4 s | 1.53× | 36.0 / 47.7 s | 1.32× | 553k / 389k |
| MNIST8M | 21.1 / 29.4 s | 1.40× | 24.9 / 35.3 s | 1.42× | 302k / 218k |
| **SOREL 4M** | 38.2 / 41.6 s | **1.09×** | 98.2 / 102.6 s | **1.04×** | 88k / 70k |

HIGGS has the largest lead for both algorithms and SOREL 4M the smallest;
MNIST8M, already in the benchmark, stays as the middle case. An earlier
screen on one socket (`sweep-one-socket/ampere-screen.csv`) also had airline
(5.4× / 4.1×) and Covertype (too short to count, 0.1 s); airline is not one
of the paper's datasets.

## Commands

`bench/tools/cluster/final.sh -k hybrid` queues `hybrid_vs_gpu_only.sh` on hopper
and ampere after their test-set runs. By hand, from the repository root after
`bench/tools/cluster/build.sh` (hopper has no git, so the version comes along;
`RUNS` defaults to 3):

```bash
for n in ampere hopper; do
  TREE_GIT_DESCRIBE="$(git describe --always --abbrev=12)" RUNS=10 \
    sbatch -J hvg-$n -p $n -w $n --exclusive -t 4-00:00:00 -o bench/hybrid-vs-gpu-only/$n.log \
      bench/tools/cluster/hybrid_vs_gpu_only.sh $n
done
```

Set the variables in the environment, not with `sbatch --export=...` (on
mordor3 that can hold the job). `DATASETS`, `PROTOCOLS` and
`PROFILE_DATASETS` narrow a run.

Charts and the profile tables:

```bash
d=bench/hybrid-vs-gpu-only
for n in ampere hopper; do
  bench/tools/.venv/bin/python bench/tools/hybrid_chart.py $d --node $n --datasets higgs,mnist8m,sorel_4m
  bench/tools/.venv/bin/python bench/tools/hybrid_profile_chart.py $d --node $n --dataset mnist8m
done
bench/tools/.venv/bin/python bench/tools/hybrid_profile_chart.py $d --node ampere --dataset mnist8m --hybrid-only --seconds
bench/tools/.venv/bin/python bench/tools/hybrid_profile_chart.py $d --node ampere --dataset mnist8m --hybrid-only
```
