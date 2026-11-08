# hybrid-vs-gpu-only: ./tree's CUDA hybrid against GPU only, by depth

![ampere](chart-ampere.png)

![hopper](chart-hopper.png)

**What.** ./tree `--cuda` (hybrid: a node with fewer rows than
`--gpu-min-rows` is copied to the host and its subtree grown by the CPU
threads while the GPU goes on with the next level) against `--cuda --gpu-only`
(every node, down to the leaves, on the GPU). CART and C4.5 without pruning,
grown to depth 5, 10, 15, 20, 25, 30 and without a depth limit, 3 runs a
case, on three of the paper's datasets, chosen by a screen of five
([Datasets](#datasets)): HIGGS (10.5M rows × 28 features), where the hybrid's
lead is the largest; SOREL-20M's 4M sample (4M × 2,381), where it is the
smallest; and MNIST8M (8.1M × 784) in between. The trees are the same in both
modes (node counts under each group). Until 2026-11-08 the benchmark had
airline in place of HIGGS and SOREL 4M (its rows are in git, 9af6148).

**Where.** The whole node (`--exclusive`), one GPU, every CPU of the node:
ampere, A100 80 GB PCIe (GPU 0, NUMA node 0), 48 threads (CPUs 0–23, then
24–47); hopper, H100 PCIe (GPU 2, NUMA node 1), 32 threads (16–31, then 0–15).
The GPU's socket's CPUs come first and the memory is not bound
([bench/cluster/README.md](../../cluster/README.md#settings-specific-to-the-cluster)).
./tree 95b6d77 (sm_80 and sm_90) throughout:
- HIGGS and SOREL 4M, every depth and the `--gpu-min-rows` sweep:
  2026-11-07/08, jobs 225 (ampere) and 226 (hopper), `*-higgs-sorel.log`.
- MNIST8M depth 15 to no limit and its sweep: 2026-11-06/07, jobs 174 and
  175, `*-both-sockets.log`.
- MNIST8M depth 5 and 10 are from the first runs of this benchmark, on the
  GPU's socket only (ampere 24 threads with `numactl --membind=0`, ./tree
  fef9ac0, 2026-11-03; hopper 16 threads with `--membind 1`, `src/` of
  fef9ac0, 2026-11-06, job 60), at `--gpu-min-rows 8192`; their run folders
  keep that job's `machine-depth5-10.json`, `plan-depth5-10.json` and
  `run-depth5-10.log`. They were not rerun: at depth 10 and below the hybrid
  hands almost nothing to the CPU, and a test on ampere found both sockets
  within ±6% of one there.

## Results (ampere)

Mean training time of 3 runs (presort, build; not reading the data). The
spread is at most 5% except on HIGGS, whose runs take 0.15–4 s (up to 12%).
Hybrid / GPU only, GPU only / hybrid in brackets.

| Depth | HIGGS CART | HIGGS C4.5 | MNIST8M CART | MNIST8M C4.5 | SOREL 4M CART | SOREL 4M C4.5 |
|---|---|---|---|---|---|---|
| 5 | 0.145 / 0.148 s (1.03×) | 0.152 / 0.153 s (1.01×) | 3.26 / 3.22 s (0.99×) | 3.33 / 3.30 s (0.99×) | 3.17 / 3.14 s (0.99×) | 3.72 / 3.74 s (1.01×) |
| 10 | 0.216 / 0.206 s (0.95×) | 0.215 / 0.216 s (1.00×) | 5.47 / 5.34 s (0.98×) | 5.40 / 5.53 s (1.02×) | 4.94 / 4.84 s (0.98×) | 5.26 / 5.29 s (1.01×) |
| 15 | 0.304 / 0.322 s (1.06×) | 0.295 / 0.317 s (1.07×) | 7.47 / 8.82 s (1.18×) | 7.49 / 7.98 s (1.07×) | 6.67 / 6.79 s (1.02×) | 6.79 / 7.00 s (1.03×) |
| 20 | 0.456 / 0.822 s (1.80×) | 0.389 / 0.567 s (1.46×) | 8.75 / 14.8 s (1.69×) | 9.17 / 11.6 s (1.27×) | 8.04 / 8.98 s (1.12×) | 8.38 / 8.58 s (1.02×) |
| 25 | 0.575 / 2.00 s (3.47×) | 0.486 / 1.09 s (2.25×) | 9.86 / 18.5 s (1.87×) | 10.6 / 15.6 s (1.48×) | 9.40 / 10.9 s (1.16×) | 9.90 / 10.3 s (1.04×) |
| 30 | 0.690 / 3.11 s (4.51×) | 0.542 / 1.67 s (3.08×) | 11.0 / 20.5 s (1.86×) | 12.0 / 19.2 s (1.60×) | 10.8 / 12.8 s (1.19×) | 11.4 / 12.0 s (1.05×) |
| none | 0.816 / 3.97 s (4.87×) | 0.676 / 2.88 s (4.27×) | 17.0 / 29.6 s (1.75×) | 22.1 / 35.0 s (1.58×) | 35.2 / 41.7 s (1.18×) | 95.0 / 102 s (1.08×) |

- **Up to depth 10 the modes are equal** (0.95–1.03×; HIGGS's 0.95× at
  depth 10 is 10 ms): few nodes get below `--gpu-min-rows`, so the hybrid
  hands almost nothing to the CPU, and both run the same GPU levels.
- **Deeper, the hybrid's lead follows the number of small nodes.** HIGGS's
  unlimited trees have 2.9M (CART) and 1.3M (C4.5) nodes over 69–71 levels,
  nearly all of them small: GPU only pays a whole level's kernels and
  synchronisation for every level of nodes of a few rows, where the hybrid's
  CPU threads grow those subtrees in one go, so the lead climbs from 1.5–1.8×
  at depth 20 to 4.3–4.9× without a limit. MNIST8M's trees have 0.2–0.3M
  nodes, and the lead stays at about 1.6–1.9× from depth 20 on (CART).
- **SOREL 4M barely gains (1.02–1.19×).** Its trees are small (88k CART
  nodes, 70k C4.5) and long and thin: no limit means depth 231 (CART) and
  572 (C4.5), the levels past 30 peeling a few rows a time off nodes that
  stay above `--gpu-min-rows`. Those nodes, and every level's 2,381
  features, stay on the GPU in both modes; C4.5's levels past depth 30 take
  84 of its 95 s.
- **The gap is all in the build.** GPU setup is the same in both modes
  (ampere: HIGGS 0.1 s, MNIST8M 1.2 s, SOREL 4M 1.8–2.2 s). Build at no depth limit:
  HIGGS CART 0.72 s hybrid, 3.88 s GPU only; MNIST8M CART 15.7 s, 28.4 s;
  SOREL 4M CART 33.4 s, 39.9 s.
- **VRAM:** the hybrid needs a little less (HIGGS 3.3 GB against 3.8–4.2,
  SOREL 4M 75.8 GB against 76.2–76.4 of the 80 GB).

## Results (hopper)

Mean training time of 3 runs, as for ampere. The spread is at most 5% except
on HIGGS (up to 16%) and SOREL 4M at depth 5 to 25 (up to 16%, the two modes
alike: 0.93× at depth 5 is within it). Hybrid / GPU only, GPU only / hybrid
in brackets.

| Depth | HIGGS CART | HIGGS C4.5 | MNIST8M CART | MNIST8M C4.5 | SOREL 4M CART | SOREL 4M C4.5 |
|---|---|---|---|---|---|---|
| 5 | 0.115 / 0.115 s (1.00×) | 0.125 / 0.121 s (0.97×) | 1.90 / 1.89 s (0.99×) | 2.01 / 2.00 s (1.00×) | 2.00 / 1.86 s (0.93×) | 2.40 / 2.23 s (0.93×) |
| 10 | 0.161 / 0.153 s (0.95×) | 0.172 / 0.170 s (0.99×) | 3.58 / 3.56 s (0.99×) | 3.66 / 3.71 s (1.01×) | 3.09 / 3.13 s (1.01×) | 3.46 / 3.46 s (1.00×) |
| 15 | 0.229 / 0.248 s (1.08×) | 0.230 / 0.246 s (1.07×) | 5.47 / 6.98 s (1.28×) | 5.65 / 6.21 s (1.10×) | 4.16 / 4.58 s (1.10×) | 4.51 / 4.72 s (1.05×) |
| 20 | 0.347 / 0.750 s (2.16×) | 0.300 / 0.468 s (1.56×) | 6.56 / 12.1 s (1.84×) | 6.91 / 9.34 s (1.35×) | 5.29 / 6.16 s (1.17×) | 5.82 / 5.94 s (1.02×) |
| 25 | 0.515 / 2.07 s (4.02×) | 0.375 / 0.964 s (2.57×) | 7.67 / 15.1 s (1.96×) | 8.12 / 12.7 s (1.56×) | 6.12 / 7.85 s (1.28×) | 6.69 / 6.97 s (1.04×) |
| 30 | 0.674 / 3.21 s (4.77×) | 0.451 / 1.57 s (3.49×) | 8.66 / 17.0 s (1.96×) | 9.29 / 15.8 s (1.70×) | 7.06 / 9.22 s (1.31×) | 7.83 / 8.46 s (1.08×) |
| none | 0.810 / 4.01 s (4.95×) | 0.598 / 2.72 s (4.56×) | 13.0 / 24.5 s (1.88×) | 17.3 / 29.0 s (1.67×) | 24.6 / 30.6 s (1.24×) | 67.6 / 73.3 s (1.08×) |

The trees are the same as ampere's (node counts in both charts).

**hopper against ampere.**
- **The same order:** HIGGS 4.6–5.0× at no limit, MNIST8M 1.7–1.9×, SOREL
  4M 1.08–1.24×; from depth 20 the hybrid's lead on hopper is the same as
  ampere's or a little larger.
- **Where the GPU does the work the H100 is faster:** MNIST8M depth 5
  1.90 s against 3.26 s, SOREL 4M no limit 24.6 / 67.6 s against 35.2 /
  95.0 s (1.4×).
- **HIGGS's hybrid at no limit is CPU bound:** 0.81 s on both nodes, with 32
  threads against 48; GPU only too is the same (4.0 s), its long tail of
  levels of a few nodes bound by launches and synchronisation, not by the GPU.

## Both sockets against one

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

## Who is working (MNIST8M, ampere, profiled)

![profile](chart-profile-ampere-mnist8m.png)

The MNIST8M depth runs once more with `./tree --profile`, 3 runs a case
(`profile/{cuda,gpuonly}-ampere-mnist8m/`): depth 15 to no limit in the same
job as the plain runs (174, 48 threads on both sockets, `--gpu-min-rows
16384`, 2026-11-07), depth 5 and 10 from the first, one-socket runs (c992b84,
24 threads, `--gpu-min-rows 8192`, 2026-11-04, jobs 87 and 88). Profiling
costs little here: every profiled mean is within 2.2% of the plain run's.

Each millisecond of training goes to one segment
([hybrid_profile_chart.py](../../hybrid_profile_chart.py)):
- **Setup:** before growing: GPU context and memory, upload, presort.
- **GPU alone:** a GPU kernel runs and no CPU thread grows a subtree.
- **GPU + CPU at once:** a kernel runs while CPU threads grow subtrees (hybrid only).
- **CPU alone:** no kernel or copy runs, CPU threads grow subtrees.
- **Transfer:** a host↔GPU copy runs with no kernel, or the host stages copies.
- **Host:** none of these: launching kernels, waiting, assembling the tree, C4.5's tidy-up.

CPU subtree work is the threads sampled (every 5 ms) in sweep, partition,
mark, build or leaves; under each bar its thread-seconds.

- **Setup is the same 1.19–1.29 s in both modes**, 38–39% of a depth-5 tree, 4–7% at no limit.
- **The CPU's subtree work grows with depth**, CART 0 → 4 → 104 → 190 → 255 → 307 → 445
  thread-seconds (depth 5 → no limit), C4.5 0 → 2 → 31 → 89 → 156 → 217 → 409;
  at no limit that is 19–28 of the 48 threads busy on average while growing.
- **It almost all runs alongside the GPU:** the GPU + CPU segment grows from 0
  to 66% (CART) and 75% (C4.5) of the hybrid's time; CPU alone, the GPU
  waiting for it or for the host between levels, stays at 0–1.4 s (most at
  C4.5 no limit, 6%, and CART depth 30, 5%). On one socket it reached 12% at
  CART depth 25–30: the second socket's threads take most of that wait away.
- **GPU only does that work on the GPU instead**: its GPU segment grows from
  1.9 s to 24.7 s (CART) and 29.9 s (C4.5), against the hybrid's 13.9 s
  and 18.6 s of GPU time (alone + with CPU). Its host time (launching and
  synchronising many short levels) grows to 2.5–3 s, 5–10% from depth 20 on,
  where the hybrid's is under 0.1 s.
- **The hybrid pays 0.5–1.6 s of transfer** (the small nodes' columns copied to the
  host) from depth 15 on, against 0.08–1.12 s for GPU only. It is larger
  than on one socket (0.3–1 s), probably because at the larger
  `--gpu-min-rows` more nodes go to the CPU.
- Up to depth 10 the two modes' bars are the same, as are their times.

The hybrid alone, in seconds on a linear axis shared by both panels, and as
shares of each bar:

![hybrid profile, seconds](chart-profile-hybrid-seconds-ampere-mnist8m.png)

![hybrid profile, percent](chart-profile-hybrid-percent-ampere-mnist8m.png)

Table (seconds, share of the training time in brackets; `hybrid_profile_chart.py` prints it):

| Algorithm | Depth | Mode | runs | train s | without --profile s | nodes | Setup (GPU init, upload, presort) s (%) | GPU working alone s (%) | GPU + CPU working at once s (%) | CPU working alone s (%) | CPU↔GPU transfer s (%) | Host: launching, waiting, finishing s (%) | CPU subtree thr·s |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CART | 5 | hybrid | 3 | 3.20 | 3.26 | 63 | 1.25 (39) | 1.90 (59) | 0.00 (0) | 0.00 (0) | 0.03 (1) | 0.02 (1) | 0 |
| CART | 5 | GPU only | 3 | 3.25 | 3.22 | 63 | 1.28 (39) | 1.94 (60) | 0.00 (0) | 0.00 (0) | 0.01 (0) | 0.03 (1) | 0 |
| CART | 10 | hybrid | 3 | 5.49 | 5.47 | 2,047 | 1.26 (23) | 3.70 (67) | 0.23 (4) | 0.00 (0) | 0.27 (5) | 0.03 (1) | 4 |
| CART | 10 | GPU only | 3 | 5.35 | 5.34 | 2,047 | 1.27 (24) | 4.02 (75) | 0.00 (0) | 0.00 (0) | 0.02 (0) | 0.05 (1) | 0 |
| CART | 15 | hybrid | 3 | 7.55 | 7.47 | 44,359 | 1.25 (17) | 2.80 (37) | 2.14 (28) | 0.09 (1) | 1.25 (17) | 0.03 (0) | 104 |
| CART | 15 | GPU only | 3 | 8.78 | 8.82 | 44,359 | 1.20 (14) | 7.01 (80) | 0.00 (0) | 0.00 (0) | 0.18 (2) | 0.40 (5) | 0 |
| CART | 20 | hybrid | 3 | 8.80 | 8.75 | 168,241 | 1.25 (14) | 2.59 (29) | 3.37 (38) | 0.17 (2) | 1.40 (16) | 0.04 (0) | 190 |
| CART | 20 | GPU only | 3 | 14.74 | 14.78 | 168,241 | 1.26 (9) | 11.51 (78) | 0.00 (0) | 0.00 (0) | 0.60 (4) | 1.37 (9) | 0 |
| CART | 25 | hybrid | 3 | 9.88 | 9.86 | 231,991 | 1.22 (12) | 2.39 (24) | 4.51 (46) | 0.29 (3) | 1.44 (15) | 0.03 (0) | 255 |
| CART | 25 | GPU only | 3 | 18.41 | 18.45 | 231,991 | 1.20 (7) | 14.40 (78) | 0.00 (0) | 0.00 (0) | 0.89 (5) | 1.92 (10) | 0 |
| CART | 30 | hybrid | 3 | 11.00 | 11.02 | 257,021 | 1.21 (11) | 2.39 (22) | 5.40 (49) | 0.50 (5) | 1.48 (13) | 0.03 (0) | 307 |
| CART | 30 | GPU only | 3 | 20.56 | 20.45 | 257,021 | 1.25 (6) | 16.31 (79) | 0.00 (0) | 0.00 (0) | 0.93 (5) | 2.07 (10) | 0 |
| CART | none | hybrid | 3 | 17.03 | 16.95 | 301,939 | 1.26 (7) | 2.65 (16) | 11.22 (66) | 0.28 (2) | 1.58 (9) | 0.05 (0) | 445 |
| CART | none | GPU only | 3 | 29.57 | 29.63 | 301,939 | 1.28 (4) | 24.71 (84) | 0.00 (0) | 0.00 (0) | 1.07 (4) | 2.52 (9) | 0 |
| C4.5 | 5 | hybrid | 3 | 3.38 | 3.33 | 61 | 1.28 (38) | 2.01 (59) | 0.01 (0) | 0.00 (0) | 0.05 (1) | 0.03 (1) | 0 |
| C4.5 | 5 | GPU only | 3 | 3.33 | 3.30 | 61 | 1.27 (38) | 2.02 (61) | 0.00 (0) | 0.00 (0) | 0.01 (0) | 0.03 (1) | 0 |
| C4.5 | 10 | hybrid | 3 | 5.52 | 5.40 | 1,453 | 1.27 (23) | 3.78 (68) | 0.38 (7) | 0.00 (0) | 0.06 (1) | 0.04 (1) | 2 |
| C4.5 | 10 | GPU only | 3 | 5.57 | 5.53 | 1,453 | 1.29 (23) | 4.22 (76) | 0.00 (0) | 0.00 (0) | 0.02 (0) | 0.05 (1) | 0 |
| C4.5 | 15 | hybrid | 3 | 7.50 | 7.49 | 12,313 | 1.21 (16) | 3.40 (45) | 2.37 (32) | 0.01 (0) | 0.48 (6) | 0.04 (1) | 31 |
| C4.5 | 15 | GPU only | 3 | 8.03 | 7.98 | 12,313 | 1.23 (15) | 6.52 (81) | 0.00 (0) | 0.00 (0) | 0.08 (1) | 0.20 (3) | 0 |
| C4.5 | 20 | hybrid | 3 | 9.19 | 9.17 | 42,915 | 1.21 (13) | 2.58 (28) | 4.63 (50) | 0.02 (0) | 0.70 (8) | 0.04 (0) | 89 |
| C4.5 | 20 | GPU only | 3 | 11.62 | 11.62 | 42,915 | 1.19 (10) | 9.57 (82) | 0.00 (0) | 0.00 (0) | 0.26 (2) | 0.60 (5) | 0 |
| C4.5 | 25 | hybrid | 3 | 10.62 | 10.58 | 84,917 | 1.22 (11) | 2.39 (23) | 6.08 (57) | 0.05 (0) | 0.84 (8) | 0.04 (0) | 156 |
| C4.5 | 25 | GPU only | 3 | 15.60 | 15.64 | 84,917 | 1.24 (8) | 12.80 (82) | 0.00 (0) | 0.00 (0) | 0.42 (3) | 1.15 (7) | 0 |
| C4.5 | 30 | hybrid | 3 | 11.96 | 11.97 | 123,289 | 1.19 (10) | 2.20 (18) | 7.40 (62) | 0.15 (1) | 0.96 (8) | 0.05 (0) | 217 |
| C4.5 | 30 | GPU only | 3 | 19.19 | 19.19 | 123,289 | 1.22 (6) | 15.72 (82) | 0.00 (0) | 0.00 (0) | 0.59 (3) | 1.67 (9) | 0 |
| C4.5 | none | hybrid | 3 | 22.42 | 22.11 | 218,325 | 1.23 (5) | 1.78 (8) | 16.84 (75) | 1.39 (6) | 1.12 (5) | 0.07 (0) | 409 |
| C4.5 | none | GPU only | 3 | 35.24 | 35.03 | 218,325 | 1.24 (4) | 29.88 (85) | 0.00 (0) | 0.00 (0) | 1.12 (3) | 3.00 (9) | 0 |

## --gpu-min-rows: the sweep

Before the depth runs, each job runs the hybrid at no depth limit, CART and
C4.5, at ./tree's default (CART rows/8192 clamped to 1,024–2,048, C4.5
rows/4096 clamped to 1,024–4,096) and at fixed values, 3 runs a value, not
interleaved (`hybrid_vs_gpu_only.sh`). The value taken has the smallest mean,
over CART and C4.5, of its time over that algorithm's fastest. Results:
`sweep/cuda-<node>-<dataset>-mr<value>/`, summary `sweep/<node>-<dataset>.json`.

ampere (48 threads):

| HIGGS | default | 256 | **512** | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 0.835 | 1.03 | **0.813** | 0.822 | 0.864 | 0.973 | 1.08 | 1.29 | 1.49 |
| C4.5 (s) | 0.684 | 0.835 | **0.699** | 0.700 | 0.683 | 0.741 | 0.824 | 1.01 | 1.23 |

| MNIST8M | default | 256 | 512 | 1024 | 2048 | 4096 | 8192 | **16384** | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 21.0 | 23.2 | 22.1 | 21.1 | 20.1 | 19.4 | 18.3 | **17.0** | 20.4 |
| C4.5 (s) | 24.8 | 29.0 | 27.3 | 26.0 | 24.8 | 23.7 | 22.5 | **22.2** | 21.0 |

| SOREL 4M | default | 256 | 512 | 1024 | 2048 | 4096 | 8192 | **16384** | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 38.4 | 40.0 | 39.1 | 38.4 | 37.7 | 36.8 | 36.2 | **35.2** | 45.8 |
| C4.5 (s) | 98.3 | 99.0 | 98.8 | 98.4 | 97.7 | 97.0 | 96.1 | **95.1** | 94.9 |

hopper (32 threads):

| HIGGS | default | 256 | **512** | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 0.828 | 0.823 | **0.811** | 0.826 | 0.857 | 0.926 | 1.08 | 1.25 | 1.43 |
| C4.5 (s) | 0.645 | 0.610 | **0.597** | 0.603 | 0.635 | 0.685 | 0.799 | 0.929 | 1.14 |

| MNIST8M | default | 256 | 512 | 1024 | 2048 | 4096 | 8192 | **16384** | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 16.7 | 19.0 | 17.6 | 16.8 | 16.1 | 15.4 | 14.3 | **13.0** | 16.0 |
| C4.5 (s) | 20.0 | 23.7 | 22.4 | 21.0 | 20.1 | 18.9 | 18.0 | **17.5** | 16.1 |

| SOREL 4M | default | 256 | 512 | 1024 | 2048 | 4096 | 8192 | **16384** | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 27.6 | 28.7 | 28.0 | 27.5 | 27.2 | 26.1 | 25.3 | **24.7** | 30.9 |
| C4.5 (s) | 70.2 | 70.8 | 70.4 | 70.1 | 69.9 | 69.3 | 68.4 | **67.8** | 70.5 |

- **HIGGS takes 512 on both nodes**, and its minimum is flat: ./tree's
  default (1,281 / 2,563) is within 8% of it (C4.5 on ampere 2% faster).
- **MNIST8M and SOREL 4M take 16,384 on both nodes.** MNIST8M's CART is
  24–28% and its C4.5 12–14% faster than the default (1,024 / 1,977); C4.5
  alone would take 32,768 (6–9% faster than 16,384), where CART is 20–23%
  slower. SOREL 4M moves little (CART 8–11%, C4.5 3–4%), as most of its work
  never reaches `--gpu-min-rows`.

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

## Datasets

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

HIGGS and SOREL 4M, one job per node, through a wrapper (not kept) that ran
`bench/cluster/hybrid_vs_gpu_only.sh` with `DIR=hybrid-vs-gpu-only/new-datasets`,
`DATASETS="higgs sorel_4m"`, `PROFILE_DATASETS=""` and
`TREE_CUDA_BIN=~/dt/tree_95b6d774_<node>` (the binary of the MNIST8M runs),
merged here afterwards. Without the wrapper, from the repository root after
`bench/cluster/build.sh`:

```bash
for n in ampere hopper; do
  sbatch -J hvg-$n -p $n -w $n --exclusive -t 16:00:00 -o bench/results/hybrid-vs-gpu-only/$n-higgs-sorel.log \
      --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty)",DATASETS="higgs sorel_4m",PROFILE_DATASETS= \
      bench/cluster/hybrid_vs_gpu_only.sh $n
done
```

MNIST8M depth 15 to no limit and its sweep (jobs 174, 175; the profile on ampere only):

```bash
P=cart_depth15,cart_depth20,cart_depth25,cart_depth30,cart_full_nodepth,c45_depth15_unpruned,c45_depth20_unpruned,c45_depth25_unpruned,c45_depth30_unpruned,c45_unpruned
sbatch -J dt-hvg -p ampere -w ampere --exclusive -t 14:00:00 -o bench/results/hybrid-vs-gpu-only/ampere-both-sockets.log \
    --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty)",PROTOCOLS=$P,DATASETS=mnist8m,PROFILE_DATASETS=mnist8m \
    bench/cluster/hybrid_vs_gpu_only.sh ampere
sbatch -J dt-hvg -p hopper -w hopper --exclusive -t 10:00:00 -o bench/results/hybrid-vs-gpu-only/hopper-both-sockets.log \
    --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty)",PROTOCOLS=$P,DATASETS=mnist8m,PROFILE_DATASETS= \
    bench/cluster/hybrid_vs_gpu_only.sh hopper
```

(The `*-both-sockets.log` jobs also ran airline, removed here; job.sh gives
the CUDA runs every CPU of the node since 95b6d77. The first runs, with the
GPU's socket only, were ampere by hand through `tune.sh` and job.sh, and
hopper as job 60, `hopper-one-socket.log`.)

Charts (`--node hopper` for hopper's) and the profile charts and table:

```bash
d=bench/results/hybrid-vs-gpu-only
bench/.venv/bin/python bench/hybrid_chart.py $d --node ampere --datasets higgs,mnist8m,sorel_4m
bench/.venv/bin/python bench/hybrid_profile_chart.py $d --node ampere --dataset mnist8m
bench/.venv/bin/python bench/hybrid_profile_chart.py $d --node ampere --dataset mnist8m --hybrid-only --seconds
bench/.venv/bin/python bench/hybrid_profile_chart.py $d --node ampere --dataset mnist8m --hybrid-only
```
