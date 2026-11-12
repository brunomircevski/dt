# Performance

What changed `./tree`'s speed and why, newest first: the numbers are from the commit each section names, so they are not current results. Those are in [bench/results/](../bench/results/README.md).

## November 2026: one exact sweep on every GPU

The CUDA grower had two ways to score cuts, picked from the GPU's FP32:FP64
ratio: one exact double-precision sweep (H100, A100: every result so far), or
a single-precision estimate of every cut followed by the exact gain of the
cuts within a proven error bound of the node's best (two-pass, for consumer
GPUs). Measured on the cluster with `--gpu-sweep` forcing each mode (main at
1e65506, 7 interleaved runs per case, `--no-prune -d 10`, hybrid and
`--gpu-only`; build time median, ms; the profiles' sweep kernels agree):

| Dataset | Algorithm | H100 one-pass | H100 two-pass | A100 one-pass | A100 two-pass |
|---|---|---:|---:|---:|---:|
| HIGGS | CART | 80 | 76 (−5%) | 113 | 105 (−7%) |
| HIGGS | C4.5 | 85 | 80 (−7%) | 117 | 110 (−5%) |
| MNIST8M | CART | 3086 | 3009 (−3%) | 3948 | 3847 (−3%) |
| MNIST8M | C4.5 | 3196 | 3567 (+12%) | 4076 | 4367 (+7%) |
| SOREL 4M | CART | 2107 | 2073 (−2%) | 2952 | 2876 (−3%) |
| SOREL 4M | C4.5 | 2235 | 2204 (−1%) | 3149 | 3078 (−2%) |
| ECBDL'14 15M | CART | 2259 | 2231 (−1%) | 3192 | 3114 (−2%) |
| ECBDL'14 15M | C4.5 | 2273 | 2242 (−1%) | 3227 | 3142 (−3%) |

(Hybrid; `--gpu-only` is within 1% of it.) The float pass saves 30–70 ms
of sweep kernels on the big two-class sets, but it keeps candidates only for
up to 8 classes: on MNIST8M (10 classes) the exact pass sweeps the surviving
tiles again, which costs C4.5 210–440 ms. Over all 16 cases one-pass was 1.3%
faster on the H100 and 0.4% slower on the A100. Two-pass was removed (its
kernels, buffers and `--gpu-sweep`), so every GPU runs the same, simpler
code.

## November 2026: the CPU sweeps and partitions after the grower split (mordor3 cluster)

After a4d89adf (cpu_builder.cpp split into presort, cpu_sweep and
cpu_builder), one-thread C4.5 on HIGGS built in 19.1 s on dell against
16.6 s before (ebbea71); epyc and CART did not move. `--profile` put the
whole loss in the partition half of `partition_sweep` (8.65 → 11.33 s; the
sweeps, 7.6 s, unchanged). The two-class sweep compiled to the same code. But
`partitionAndSweep<Entropy, true>` grew from 2490 to 3382 bytes: with
`sweepSetup()` now out of line, GCC 13 inlined the staging buffers'
`flush()` into it. Its loop already kept the two buffers' sizes in memory
(a store and a reload per entry), and with the inlined flush that cost the
3 s on the Xeon 6542Y (no PMU counters there, so the exact stall is not
known). Marking `flush()` noinline gave back 16.5 s. The fix that went in
does not depend on what GCC inlines: entries go to the buffers in runs that
cannot fill one, with the sizes in registers, and full buffers are flushed
between runs (8dedb431). MNIST8M's one-thread C4.5 on epyc was also slower
after the split (sweep 48 → 60 s in `--profile`) with the same instructions in
`scanFeatureK`, so most likely code or data placement; the changes below
replace that code.

Every change keeps the trees: the `--dump` of every run below has the same
SHA-256 as ec77727's, `tests/check.sh` passes for `./tree_cpu` (dell3) and
`./tree` with the CUDA backend (hopper), and the CUDA build compiles for
sm_75/80/90. In the order of the commits, one-thread build on dell's HIGGS
C4.5 (interleaved runs in separate jobs):

* **Staging runs** (8dedb431): 19.17 → 16.10 s, back to ebbea71's 16.59 s.
* **Blocks without a threshold** (f94a9158). C4.5's 1e-5 gap makes most
  16-cut blocks of HIGGS's sweeps hold no allowed threshold; the values are
  sorted, so first-to-last of the block decides it, before the corner bound
  is computed: 16.10 → 9.92 s (SUSY C4.5 2.95 → 2.72 s). CART has no gap and
  is unaffected.
* **No branch on the row's side** in the out-of-place partition (41c2f03d,
  `moveRows()`, shared with `partitionInBlocks`): GCC made `isLeft ? left :
  right` a branch that an even split mispredicts on every other row; the
  target index is now a mask. With **galloping runs** in `scanFeatureK`
  (2c81e137: a run of values within the gap is crossed in doubling steps,
  so most of its rows are counted without comparisons) and **64-bit entry
  moves** (94f33242), MNIST8M one-thread build 236 → 156 s on dell and 188 →
  95 s on epyc (C4.5), CART 263 → 159 s and 205 → 97 s. HIGGS C4.5 on dell
  within noise of the step before; on epyc 8.9 s after the staging runs →
  6.6 s with the gap skip and these three.
* **A full block's class-1 count in a fixed-count loop** (f49ab9d4, which GCC
  vectorizes) and **four counter lanes for the rows of a run** (c444d72c):
  dell HIGGS C4.5 10.1 → 8.8 s, epyc 6.6 → 5.5 s, MNIST8M epyc C4.5 95 →
  83 s.

Final A/B, `--no-prune -d 10`, build ms, interleaved in one job per node
(HIGGS/SUSY 4 runs, MNIST8M/ECBDL 15M 3, ECBDL 15M one-thread 1); one thread
is `taskset -c 0 --serial`, 48 is `--parallel --threads 48`; CV ≤ 1.3% for
one thread, ≤ 12% at 48 threads (SUSY):

Build seconds; the change is against f1fe270.

| case | threads | dell ec77727 | dell f1fe270 | dell c45-perf | epyc ec77727 | epyc f1fe270 | epyc c45-perf |
|---|--:|--:|--:|--:|--:|--:|--:|
| HIGGS C4.5 | 1 | 16.60 | 19.13 | **8.75** (-54%) | 9.97 | 10.11 | **5.54** (-45%) |
| HIGGS CART | 1 | 12.18 | 12.15 | **9.35** (-23%) | 8.62 | 8.58 | **5.94** (-31%) |
| SUSY C4.5 | 1 | 3.10 | 3.14 | **2.57** (-18%) | 2.42 | 2.48 | **1.79** (-28%) |
| SUSY CART | 1 | 2.94 | 2.92 | **2.68** (-8%) | 2.31 | 2.33 | **1.93** (-17%) |
| MNIST8M C4.5 | 1 | 238 | 239 | **154** (-36%) | 176 | 188 | **83.0** (-56%) |
| MNIST8M CART | 1 | 266 | 266 | **157** (-41%) | 204 | 205 | **87.3** (-57%) |
| ECBDL 15M C4.5 | 1 | 346 | 363 | **185** (-49%) | 235 | 235 | **108** (-54%) |
| ECBDL 15M CART | 1 | 283 | 285 | **185** (-35%) | 186 | 188 | **104** (-45%) |
| HIGGS C4.5 | 48 | 0.75 | 0.89 | **0.61** (-32%) | 0.48 | 0.45 | **0.34** (-26%) |
| HIGGS CART | 48 | 0.62 | 0.62 | **0.57** (-8%) | 0.48 | 0.47 | **0.35** (-25%) |
| SUSY C4.5 | 48 | 0.24 | 0.26 | **0.22** (-16%) | 0.19 | 0.17 | **0.16** (-7%) |
| SUSY CART | 48 | 0.25 | 0.25 | **0.23** (-6%) | 0.20 | 0.18 | **0.18** (-1%) |
| MNIST8M C4.5 | 48 | 9.88 | 9.72 | **9.38** (-3%) | 4.64 | 4.88 | **3.50** (-28%) |
| MNIST8M CART | 48 | 9.88 | 9.78 | **9.33** (-5%) | 5.15 | 5.16 | **3.54** (-31%) |
| ECBDL 15M C4.5 | 48 | 10.86 | 11.06 | **9.10** (-18%) | 6.28 | 6.30 | **3.44** (-45%) |
| ECBDL 15M CART | 48 | 9.81 | 9.87 | **9.01** (-9%) | 5.16 | 5.17 | **3.58** (-31%) |

Tried and dropped (same trees in every case):

* **`flush()` noinline** fixes the regression (16.53 s) but leaves the
  sizes in memory; superseded by the staging runs.
* **`sweepSetup()` inline again, `-falign-loops=64`**: no change.
* **Plain stores instead of non-temporal ones** in the partitions: slower on
  both nodes (dell HIGGS C4.5 10.1 → 11.4 s, epyc 6.6 → 7.0 s, MNIST8M 1-4%).
* **Bounding growing spans of blocks** (up to 256 cuts) before the 16-cut
  blocks: slower on SUSY.
* **A `scanFeatureK<10>` instantiation** for MNIST's 10 classes: slower
  (local, 6.57 vs 6.37 s on a 500K sample).
* **No zero-fill of the multi-class counters**: no change.

## November 2026: C4.5's GPU setup on SOREL 4M (CUDA hybrid)

C4.5's setup on SOREL 4M took 3.4 s on the A100 node against CART's 1.9 s
and 2.2 s with `--gpu-only`. Host-side timings showed where: the download of
the distinct values (1G values, 4 GB) took 1.5 s in the hybrid and 0.33 s
with `--gpu-only`. The hybrid started the arena's toucher (76 GB of memset on
24 threads) just before it, and both wrote to the GPU's socket at once. The
toucher now starts after the download; it still finishes early in the build
(0.05-0.08 s longer on SOREL 4M).

The features whose distinct values do not fit the device pool (most of them
on SOREL 4M, 8x its size) were also downloaded one at a time, about 2000
synchronous copies; they now go in batches (0.12-0.15 s of the setup).

Both, ampere, hybrid, `--no-prune -d 10`, 3 alternating runs each, the same
trees in every case: SOREL 4M C4.5 6.38 s → 5.33 s (the same as
`--gpu-only`), HIGGS C4.5 237 → 218 ms, SUSY C4.5 111 → 100 ms; airline
(C4.5 at depth 10 and 20, CART), chess 10M and every CART run within noise.

## November 2026: SOREL's --simple and NUMA

One-thread runs on SOREL (121 GB of values) are sensitive to where the pages
land. The first `--simple` runs (2026-10-27/28) were the only case on any
dataset where epyc was slower than dell: 14,124 s (CART) and 14,240 s (C4.5)
against 12,414 s and 12,712 s, 1.1×; elsewhere epyc's `--simple` is 0.89–1.00×
dell. Those runs started with 159–185 GB free on epyc's node 0 (normally about
260 GB): run.py's data check still kept about 135 GiB of pyarrow memory in the
harness process, and the loader kept the 136 GB CSV in the page cache while
parsing it. ./tree ran on CPU 0, so part of the columns most likely went to
node 1, the other socket (the runs recorded no page placement, so this is not
proven). `--simple` sorts row indices by `value(row, feature)`, a chain of
random reads into a 50 MB column that is bound by memory latency, and remote
memory hurts that most; `--serial` streams presorted columns, the prefetcher
hides the latency, and its epyc runs (with the fixed harness) were 1.5× faster
than dell's, as on other data.

Fixes: the data check runs in a child process (8bdf641), and the loader drops
the CSV from the page cache as pass 2 parses it (da1f100), so a one-thread
SOREL load needs about max(136, 121) GB on its node instead of about 257. The
rerun on 32cf657 (2026-11-02, 261 GB free on node 0) took 12,501 s and
12,906 s on epyc, 11% and 9% less, and 13,325 s and 13,355 s on dell (5–7%
more than its first runs, one run each): epyc is now 0.94–0.97× dell.

## October 2026: what the benchmarks showed (mordor3 cluster)

Probes on ec77727 after the full rerun, 2026-10-30/31: the same binaries, in
alternating rounds on whole nodes. Their raw runs were not kept.

**Repeats.** Of 52 cases with under 10 runs under 10 minutes, 2 have a CV above
2% (the highest 2.9%), so more runs would not change the means. The exception
is a run with two modes: 3 fast runs hid the SOREL slowdown below, which hit 13
of 24. Run 8 or more on the big datasets after a change to the presort or to
allocation, and look at `--profile`'s idle time.

**dell at 2–12 threads is noisy, and the cause is unknown.** HIGGS at 4 threads
has a CV of 13–17% over 16 runs (epyc 0.3%); up to 48 threads and on one socket
the means hold, with a CV of 2–4% and 5–9%. Ruled out:

* **NUMA balancing:** `numactl --membind` on every node takes the migrations to
  0, with the same trees, the times within about 1% and the noise unchanged.
* **Pinning:** the first 2 of 4 rounds looked like the cure (CV 17% → 4%), but
  over all 4 pinned is worse (CV 26%): pinned runs get a slow mode of their own,
  1.5–1.65× the time at 4–12 threads. On epyc pinning changes nothing, except
  about 7% better at 48 threads on HIGGS.
* **Frequency:** a sampler at 200 ms shows the slow runs lose CPU time, not
  clock speed: user time stays flat and a CPU sits idle at 800 MHz. With
  spinners on the other 44 CPUs (3.6 GHz throughout) the CV falls from 9–11% to
  5–6%, but the mean is slower. Pinned slow runs differ: their user time rises.

So take small differences at low thread counts on dell as noise, use epyc for
them, and do A/B runs interleaved in one job. `perf` is unusable there
(`perf_event_paranoid` 4) and the governor is the admins'.

**Thread placement.** Threads take CPUs in `lscpu` order: dell's go round its 4
NUMA nodes (spread), epyc's fill the first 24 CPUs (compact), so the two curves
of `cluster-cpu-scaling` are not like for like. Compact is faster on both at low
thread counts, and equal at 48:

| HIGGS CART (s) | 2 threads | 4 | 24 |
|---|--:|--:|--:|
| dell spread | 9.0–10.6 | 5.6–5.7 | 1.22 |
| dell compact | 7.9 | 4.5–5.1 | 1.08–1.10 |
| epyc spread | 6.9 | 4.0 | 0.92 |
| epyc compact | 5.5 | 2.85 | 0.83 |

**Nodes.** dell2–4 have half dell1's memory channels and are 1.13× slower on
SUSY, 1.14× on HIGGS, 1.10–1.17× on airline, 1.30–1.34× on ECBDL 15M and
1.38–1.46× on MNIST8M, with the same single-thread speed; never mix them. epyc1
and epyc2 are within 2%. dell's single thread is 1.47× slower than epyc's on
HIGGS and 1.6× on chess 10M, but equal on chess 1000M: core speed on small data,
memory on big data.

**`--serial` on chess** costs 2.9× more per row from 10M to 1000M rows (34 → 99
ns: mark 5.6 → 41, partition 8 → 37), against 1.5× for `--simple` (617 → 912
ns). That is why its lead over `--simple` is smaller on the big sets.

**The A100 beats the H100 at chess 10M** because its fixed cost is lower (about
0.05 s against 0.07 s, 71–79% of its time). The H100 wins from 100M rows up.

**Depth.** CPU and both GPUs built identical trees at every depth below,
unlimited included. Depth 10 underfits, and it is C4.5's limit that makes it look weak:
gain ratio needs more levels, and unlimited it beats CART on all three.

| Test accuracy | depth 5 | 10 | 15 | 20 | unlimited (depth reached) |
|---|--:|--:|--:|--:|--:|
| HIGGS CART | 66.4% | 70.5% | 72.0% | 71.6% | 67.4% (69) |
| HIGGS C4.5 | 60.5% | 63.7% | 67.3% | 68.9% | 69.8% (71) |
| airline CART | 61.9% | 66.6% | 69.4% | 70.9% | 65.1% (66) |
| airline C4.5 | 61.8% | 66.2% | 68.8% | 70.4% | 68.4% (73) |
| MNIST8M CART | 61.7% | 84.5% | 91.1% | 92.6% | 92.7% (115) |
| MNIST8M C4.5 | 45.6% | 74.7% | 82.0% | 87.6% | 92.9% (147) |

The GPU's lead shrinks with depth, because small nodes go to the CPU:

| Training time (s), CART: epyc / H100 hybrid / H100 `--gpu-only` | depth 5 | 10 | 15 | 20 | unlimited |
|---|--:|--:|--:|--:|--:|
| HIGGS | 0.31 / 0.11 / 0.10 | 0.62 / 0.15 / 0.15 | 0.82 / 0.22 / 0.24 | 1.04 / 0.43 / 0.74 | 1.41 / 1.26 / 3.93 |
| airline | 1.82 / 0.32 / 0.32 | 2.87 / 0.49 / 0.49 | 4.15 / 0.69 / 0.70 | 5.06 / 0.98 / 1.33 | 6.30 / 4.51 / 21.67 |
| MNIST8M | 3.44 / 1.90 / 1.89 | 5.93 / 3.57 / 3.57 | 7.99 / 5.60 / 6.68 | 9.82 / 7.32 / 11.68 | 19.78 / 16.64 / 24.06 |

Unlimited, airline's CART has 44.7M nodes. So state the depth limit next to any
accuracy or GPU figure at depth 10.

## October 2026: SOREL's C4.5 presort on epyc

The presort took 7 s in some runs and 18 s in others, with the same tree: one
thread worked alone for 11.5 s while the other 47 waited. That was the
zero-fill of C4.5's distinct values, on one thread, between the two parallel
passes of `distinctValues()` (no reclaim: `pgscan` 0; why it is only sometimes
slow is not known). The values now stay uninitialised until those passes write
them. 24 runs of each binary, alternating, on epyc1 and epyc2 (48 threads):
13 slow before, none after; mean training 34.5 s → 28.5 s, the same trees and
peak memory, and 28.5 s is what the fast runs took before.

## October 2026: a faster presort, and experiments that did not help (mordor3 cluster)

Unpruned trees (`--no-prune -d 10`); every run's `--dump` compared with the
base binary's. Two CPU presort changes are in `cpu_builder.cpp`
([CPU.md](CPU.md)): the digit histograms of big columns are counted in chunks
on all threads, and the radix sort covers only the key bits that vary
(integer columns keyed by `int(v) − min`, 12-bit digits, columns of 1M+ rows).
Same trees (`tests/check.sh`, 24 dumps). `--parallel` on dell (48 threads),
medians of 3 interleaved reps against main 8bdf641:

| | presort CART | train CART | presort C4.5 | train C4.5 |
|---|--:|--:|--:|--:|
| chess_1b | −62% | −32% | −49% | −20% |
| chess_100m | −52% | −24% | −49% | −19% |
| chess_10m | −37% | −17% | −42% | −6% |
| airline | −34% | −7% | −29% | −2% |
| ECBDL 15M | −25% | −4% | −19% | −6% |
| SUSY | −17% | −0% | −15% | −12% |
| covertype | −9% | −4% | −7% | −7% |
| mnist8m | −3% | −2% | −8% | +3% |
| HIGGS | +3% | +1% | +3% | −2% |

Build time is unchanged (dell noise ±8%). HIGGS's 28 columns of 11M rows keep
all threads busy without chunks, and its float columns save no pass. Serial:
covertype / SUSY / HIGGS presort −2 / −9 / −4% (CART), train within 2%.

Tried and dropped (same trees in every case):

* **Skipping the column writes of the last level** (fused child sweeps at depth
  max−1 write columns nobody reads again): build −5% to +3%, noise.
* **Prefetching the left/right flag** 32 rows ahead in the partition loops:
  within ±3%.
* **CUDA: packed upload** (integer columns sent as uint8/16 and unpacked by a
  kernel) and **cub radix over the varying bits only**: both slower on every
  dataset, train +28 to +105% on the H100 and +3 to +40% on the A100 (e.g.
  airline CART 530 → 1089 / 983 ms on the H100, 793 → 1068 / 1054 ms on the
  A100). Both went through an upload path that scans each staged chunk on the
  host before sending it (airline user time 33 → 58 s), which costs more than
  the saved bytes or radix passes. A new try would have to fold the scan into
  the staging copy.
* **Page-locking the hand-off arena in 1 GB chunks** (2026-10-18): ECBDL CART
  on the H100 23.0 → 18.6 s, but C4.5 on the A100 66.5 → 68.6 s and airline C4.5
  crashed in chunk mode. Superseded by the staged hand-offs below, which do not
  page-lock the arena at all.
* The `--parallel` hang (about 1 in 40 big runs on dell) was a lost wake-up in
  libstdc++ 13's `atomic::notify` after a release store; `waitNonZero()` in
  `cpu_builder.cpp` avoids `atomic::wait` (a `seq_cst` store also fixed it but
  made ECBDL C4.5 7% slower).
* **The waits of `partitionInBlocks`.** Spinning on `_mm_pause` without end kept
  a descheduled thread off its CPU: on epyc chess_10m, 13 of 50 runs took over
  twice the median. `atomic::wait` after 2,048 pauses made dell's chess about 2×
  slower in some node states, and brought the hang above. `waitNonZero()` does
  2,048 pauses, then `yield` for up to 2 ms, then 50 µs sleeps, with release
  stores and no notify. On dell chess 100M and 1B got about twice as fast.
* **Code alignment.** One build was 8% slower on epyc's ECBDL 15M C4.5 (7.80 →
  8.41 s) because C4.5's sweep landed on another offset mod 64;
  `-falign-functions=64 -falign-loops=32` in the Makefile restores it.

## October 2026: hand-offs without page-locking the arena (CUDA, mordor3 cluster)

The host arena that receives the hand-offs (75 GB for ECBDL'14's 15M sample)
is no longer page-locked. Its registration held the CUDA driver's lock for
1.2–1.9 s, and 4.8–11.5 s on ampere when the node's memory was busy, and the
GPU loop waited for it. Now each hand-off piece is downloaded into one of two
page-locked 128 MB buffers (one per copy stream), and a host function copies
it into the arena on the staging threads while the other stream downloads the
next piece ([CUDA.md](CUDA.md)). Same trees. Medians against e0a9106
(A/B logs: `bench/results/cluster-handoff-staging/` at 6f6cecb):

| Train total (s) | H100 before | H100 after | A100 before | A100 after |
|---|--:|--:|--:|--:|
| ECBDL 15M C4.5 | 52.73 | 51.25 | 74.68 | 65.73 |
| ECBDL 15M CART, no pruning | 18.73 | 17.60 | 31.73 | 23.89 |
| airline C4.5 | 6.19 | 6.08 | 5.95 | 5.65 |
| airline CART, no pruning | 4.19 | 3.31 | 3.79 | 3.65 |
| HIGGS C4.5 | 1.10 | 1.24 | 1.18 | 1.12 |

* The arena is still touched on all threads during the presort: without that,
  the copies faulted its pages in, and airline's levels on the H100 took
  1.8–2.0 s instead of 1.2 s. `MADV_POPULATE_WRITE` instead of the memset holds
  the process's mmap lock and stretched the presort from 0.5 to 4.2 s when the
  node's memory was low.
* 32 MB buffers (four times as many host functions, which run one at a time)
  left the levels 0.2–0.4 s slower than 128 MB ones.
* HIGGS C4.5 on the H100 lost 0.1 s: the pruning's arranged rows now go
  through the staging buffers too.

## October 2026: four CUDA changes measured on one commit (mordor3 cluster)

67495f9 added four changes ([CUDA.md](CUDA.md)): the goes-left bitmap marks
only the smaller side of each split, the split feature's own column is
partitioned by position, each feature is presorted as soon as its column is
uploaded, and the root's class counts come from a kernel during the presort.
They were measured on one commit and machine, switched by environment
variables, depth 10 without pruning, `train total` medians of 5 runs (3 for
chess_1b and ECBDL 15M). Every tree was byte-identical to the previous code.

| Dataset | H100 CART | H100 C4.5 | A100 CART | A100 C4.5 |
|---|--:|--:|--:|--:|
| airline (115M × 13) | 655 → 522 ms | 676 → 538 ms | 978 → 798 ms | 997 → 815 ms |
| chess_1b (1G × 2) | 969 → 620 ms | 969 → 670 ms | 1488 → 1055 ms | 1485 → 1127 ms |
| ECBDL 15M | 3508 → 2979 ms | 3602 → 3003 ms | 5720 → 5059 ms | 5884 → 5132 ms |
| HIGGS (10.5M × 28) | 174 → 162 ms | 202 → 178 ms | 238 → 207 ms | 255 → 244 ms |

* Without the overlap the other three change only the build: chess_1b CART
  654 → 350 ms on the H100, airline 392 → 328 ms; ECBDL and HIGGS barely move.
  On chess_1b the smaller side alone saved 0.07 s and the split column by
  position 0.12 s.
* The overlap hides the presort behind the upload: about 0.5 s on ECBDL (H100,
  0.7 s A100) and 0.07–0.09 s on airline. On chess_1b (2 features) only half
  of it, and the upload gets slower.
* The root counts on the GPU save 0.06 s on airline and 0.12 s on chess_1b
  (H100).

## October 2026: one copy of the columns (CUDA, serial, parallel)

The columns now live in **slots**, one per feature and a few spare ones, and a
partition moves each column into a free slot instead of ping-ponging between
two full copies ([CUDA.md](CUDA.md), [CPU.md](CPU.md)). The CUDA backend also
dropped its device mirror of the host arena (the hand-offs go through a
512 MB ring) and presorts in the slots. Same trees everywhere (`tests/check.sh`
for `./tree` and `./tree_cpu`, and every run below byte-identical). Means of
3 runs (GPU) or 2 (CPU parallel; serial 1), `train total`, depth 19, no pruning, the
mordor3 nodes; before = 66bd66c.

| CUDA | H100 before | H100 after | A100 before | A100 after |
|---|--:|--:|--:|--:|
| HIGGS CART | 0.39 s, 7.2 GiB | 0.40–0.42 s, 3.4 GiB | 0.49 s, 7.2 GiB | 0.49 s, 3.3 GiB |
| HIGGS C4.5 | 0.36 s, 7.2 GiB | 0.36 s, 3.3 GiB | 0.46 s, 7.1 GiB | 0.44 s, 3.3 GiB |
| airline CART | 1.33 s, 35.0 GiB | 1.38 s, 13.3 GiB | 1.94 s, 35.0 GiB | 1.98 s, 13.3 GiB |
| airline C4.5 | 1.30 s, 35.0 GiB | 1.33 s, 13.3 GiB | 1.87 s, 34.9 GiB | 1.91 s, 13.3 GiB |

* GPU memory: 53% less on HIGGS, 62% less on airline (cuML used 1.9 and
  9.5–10 GiB in the depth-19 run of cluster-gpu-vs-others, in the git history
  before 325039f).
* Airline is 2–3% slower: the partition is 3% slower (a column waits for the
  one before it to leave its slot, 3–5 ms of 500), and the GPU sometimes
  waits for the ring's copies to the host.
* Traps found on the way: tiles taken column by column instead of node by node
  made the partition 10% slower (the dependency is per segment, so the order
  could stay); a 64-bit field in `PartitionTarget`, indexed by side, pushed it
  to local memory (+5%); a move record read by every thread, or waited for by a
  second thread, +3%; growing CUB's temp buffer in the presort, or a sync per
  feature, put the presort behind the arena's page-locking (C4.5 airline
  presort 0.1 → 0.4 s), so the arena is now registered after the presort.

| CPU (48 threads) | epyc1 before | epyc1 after | dell1 before | dell1 after |
|---|--:|--:|--:|--:|
| HIGGS CART parallel | 1.04 s, 5.7 GiB | 1.06 s, 5.6 GiB | 1.28 s | 1.27 s |
| HIGGS C4.5 parallel | 1.06 s, 5.8 GiB | 1.01 s, 5.7 GiB | 1.50 s | 1.48 s |
| airline CART parallel | 7.05 s, 29.0 GiB | 7.09 s, 27.3 GiB | 7.81 s | 7.53 s |
| airline C4.5 parallel | 8.26 s, 29.0 GiB | 8.12 s, 27.3 GiB | 8.81 s | 8.94 s |
| HIGGS CART serial | 18.1 s, 5.7 GiB | 18.0 s, 3.6 GiB | 25.0 s | 24.8 s |
| HIGGS C4.5 serial | 22.3 s, 5.7 GiB | 20.7 s, 3.6 GiB | 34.1 s | 32.6 s |

* `--serial` uses one spare slot: 37% less memory on HIGGS, and 0–7% faster
  (the winner's column is no longer copied).
* `--parallel` keeps `min(F − 1, threads)` spare slots by default, so that a
  big node moves all its columns at once, one per thread, with the children's
  sweeps fused in: memory drops only where the features outnumber the threads
  (airline: 6%; ECBDL's 631 features on 48 threads need 678 slots instead of
  1262). (`--spare-slots N`, which traded speed for memory, was removed in
  November 2026.)
* The cluster benchmarks were rerun on cd18717 (2026-10-17,
  [bench/results/](../bench/results/README.md)): the same trees everywhere,
  times within a few percent of the previous runs. Full ECBDL'14 on the CPU
  servers needs 237 GiB instead of 375 GiB; its GPU sample grew from 6M to 15M
  rows, which fill 73.5 GiB of the 80 GB GPUs as 6M did before. A same-job
  comparison on the A100 (old and new binary interleaved, 5 runs): pruned
  HIGGS CART 0.50 → 0.51 s, C4.5 1.19 → 1.16 s, airline CART 1.60 → 1.63 s,
  C4.5 5.91 → 5.90 s.

## October 2026: the serial backend runs the parallel algorithm

Until 2026-10-17 `--serial` kept no second copy of the columns, so it
partitioned every node in place, through a byte per row, while `--parallel`
partitioned big nodes (≥ 32768 rows) out of place through a bitmap and swept
the big children during the partition. Now both do the latter (commit
4b1ced0); serial differs only in running on one thread (plain bit operations
instead of atomics). Same trees everywhere. One-thread `./tree` rows of the
benchmarks were rerun (median of 3, the same nodes):

| Serial | epyc1 | dell1 | desktop |
|---|--:|--:|--:|
| airline CART | 148.5 → 76.4 s (1.94×) | 204.7 → 128.1 s (1.60×) | – |
| airline C4.5 | 250.6 → 178.6 s (1.40×) | 338.3 → 264.1 s (1.28×) | – |
| HIGGS CART | 15.3 → 18.2 s (0.84×) | 29.3 → 25.7 s (1.14×) | – |
| HIGGS C4.5 | 30.5 → 37.8 s (0.81×) | 54.4 → 54.0 s (1.01×) | – |
| SUSY CART | 4.68 → 4.98 s (0.94×) | 7.97 → 6.84 s (1.16×) | 4.63 → 4.53 s (1.02×) |
| SUSY C4.5 | 5.83 → 6.77 s (0.86×) | 10.75 → 9.06 s (1.19×) | 5.99 → 5.90 s (1.01×) |

* Airline (115M rows) gains most: the old byte per row (115 MB) misses the
  caches, the bitmap (14 MB) does not.
* On epyc, HIGGS and SUSY got 6–19% slower (HIGGS CART's build 13.55 →
  16.25 s; three runs within 0.05 s). Not investigated; one candidate is the
  non-temporal stores of the out-of-place partition, tuned for many threads.
* Memory: a serial run now holds the second copy too (`8 · F · n` bytes):
  HIGGS 3.7 → 5.7 GiB, airline 20.4 → 29.0 GiB.

## October 2026: profiling airline with Nsight Systems (mordor3 cluster)

Commit `35cf315` (before) against this work (after), on airline, same nodes
and method as below: before and after in the same job, interleaved, 3 runs
each, medians of `train total` (presort + build + prune for C4.5, presort +
build for CART in brackets; on the GPU nodes the total also counts the GPU
setup). Every tree is byte-identical, and
`tests/check.sh` passes for `./tree` and `./tree_cpu`.

| Algorithm | Node, backend | Before | After | Speed-up |
|-----------|---------------|-------:|------:|--------:|
| C4.5 | hopper cuda | 7.29 s (0.54 + 3.29 + 3.22) | 5.19 s (0.09 + 2.88 + 2.02) | 1.40× |
| C4.5 | ampere cuda | 8.48 s (0.70 + 3.34 + 3.89) | 5.85 s (0.11 + 3.30 + 2.12) | 1.45× |
| C4.5 | epyc1 parallel | 12.34 s (0.72 + 10.28 + 1.36) | 11.87 s (0.47 + 10.09 + 1.33) | 1.04× |
| CART | hopper cuda | 1.18 s (0.09 + 0.92) | 1.11 s (0.09 + 0.81) | 1.06× |
| CART | ampere cuda | 1.89 s (0.10 + 1.23) | 1.59 s (0.10 + 1.17) | 1.19× |
| CART | epyc1 parallel | 7.31 s (0.67 + 6.64) | 7.11 s (0.45 + 6.66) | 1.03× |

GPU setup (upload) on ampere went from 0.56 to 0.32 s; on hopper it stays
at about 0.2 s. The time was found with Nsight Systems (`make NVTX=1`, see
below) and `perf` on ampere. What changed:
- **C4.5's distinct values on the device:** the GPU presort used to copy all
  sorted values back for the thresholds; now CUB's `DeviceSelect::Unique`
  keeps each feature's distinct values and only those are copied (presort
  0.54 → 0.09 s on hopper, 0.70 → 0.11 s on ampere). The CPU grower takes
  them from its sorted columns in parallel.
- **Copies on the GPU's NUMA node:** ampere has two sockets and its A100 on
  the first. Staging buffers and copy threads on the other socket ran at
  21 GB/s up and 16 GB/s down, against 25 and 24 on the GPU's own (hopper:
  about 55 GB/s either way). The staging buffers and a pool of copy threads
  are now pinned to the GPU's node (read from sysfs), for the upload, the
  leaves of the rows and the pruning's arrangement.
- **Asynchronous hand-off:** the level loop used to wait for each hand-off's
  copy to the host. With room on the device (airline on both GPUs) the
  columns are gathered into a device arena, copied on a second stream, and
  a host callback starts the CPU tasks; the GPU goes on with the next level.
  Buffer resizes use stream-ordered allocation, so they do not synchronize.
- **The pruning's first arrangement on the GPU:** C4.5 pruning starts by
  sorting the 115M rows by leaf and transposing their features to rows
  (0.4 s on 32-48 threads). When the device can keep a copy of the raw
  values, a radix sort and a gather kernel do it on the GPU and the result
  is copied into the pruning's workspace.
- **Leaf ids renumbered lazily:** the leaf of every row went through two
  renumberings (node store to tree, collapse of useless splits), each a pass
  over 115M ids. Now only a map from old to new ids (one per node) is
  composed, and the ids are read through it.
- **Parallel presort for few features:** with fewer features than threads
  each big column's radix sort is split across threads (digit counts per
  part, prefix over parts, stable parallel scatter): epyc1's presort 0.67 →
  0.45 s. Root class counts are counted in parallel too.

Tried and dropped:
- **Routing the pruning's extra rows on the GPU** (subtree raising counts a
  sibling's rows in the raised branch): the per-call uploads, the scan of
  changed nodes and the calls serialized on one device made the big routes
  slower (0.85 → 2.4 s on hopper).
- **More levels on the GPU** (`--gpu-min-rows` 512 or 1024 instead of 4096
  for C4.5 and 2048 for CART): slower on both GPU nodes (ampere C4.5 build
  3.3 → 5.9 s at 512). A level of many small nodes costs the GPU nearly as
  much as one of big nodes, and the CPU threads finish small subtrees fast.
  The defaults are best on hopper and on ampere, so the benchmarks need no
  per-node settings.
- **Tile sizes** of 4 or 16 entries per thread instead of 8: slower.

The CPU builder's sweep and partition (most of epyc1's 10 s C4.5 build) are
unchanged.

### Profiling with Nsight Systems

`make NVTX=1` builds `./tree` with NVTX ranges around the phases (load,
upload, presort, every GPU level, hand-off copies, CPU subtrees, merge,
pruning steps). Then, on a GPU node (not the
login node):

```
nsys profile -t cuda,nvtx,osrt -o airline-c45 ./tree --cuda --c45 train.csv
nsys stats -r nvtx_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum airline-c45.nsys-rep
```

CPU sampling (`-s cpu`) needs `perf_event_paranoid` ≤ 2: on mordor3 only
ampere allows it. Without NVTX=1 the ranges compile to nothing.

## October 2026: C4.5 pruning, the GPU partition and the upload (mordor3 cluster)

Commit `93920b9` (before) against the merge of this work (after), on airline
(115M rows, 18 features), measured on hopper (H100 PCIe + 32 Xeon cores,
`--cuda`), ampere (A100 80 GB + 48 EPYC 7402 cores, `--cuda`) and epyc1
(2× EPYC 9255, 48 cores, `--parallel`). Before and after ran in the same job,
interleaved, 3 runs each; times are medians of `train total` (build and prune
in brackets). Every tree is byte-identical before and after, on all three
nodes.

| Algorithm | Node, backend | Before | After | Speed-up |
|-----------|---------------|-------:|------:|--------:|
| C4.5 | hopper cuda | 11.08 s (3.61 + 6.66) | 7.11 s (3.30 + 3.09) | 1.56× |
| C4.5 | ampere cuda | 11.72 s (3.89 + 6.65) | 8.59 s (3.39 + 3.88) | 1.37× |
| C4.5 | epyc1 parallel | 13.78 s (10.16 + 2.90) | 12.33 s (10.26 + 1.37) | 1.12× |
| CART | hopper cuda | 2.17 s (1.72) | 1.21 s (0.92) | 1.79× |
| CART | ampere cuda | 2.88 s (2.28) | 1.86 s (1.24) | 1.55× |
| CART | epyc1 parallel | 7.36 s (6.66) | 7.40 s (6.69) | – |

CART is `--cart --alpha 1e-5`. What changed:
- **The growers record each row's leaf.** C4.5 pruning needed every row's
  leaf and routed all 115M rows through the 19.6M-node tree for it. The
  builders know a node's rows when
  it becomes a leaf, and the ids now follow the tree through its renumbering
  and the collapse of useless splits.
- **C4.5 pruning without blocking joins:** a node is settled by whichever of
  its branches finishes last, so no thread waits. The rows are grouped by
  leaf with a radix sort whose passes have equal digits (two 12-bit passes
  for airline's 4.3M leaves instead of 11 + 11 + 1 bits), and the counts of a
  sibling's rows per leaf go through a small per-thread cache.
- **GPU goes-left flags as a bitmap:** one bit per row (14 MB for airline,
  in L2) instead of a byte (115 MB). The partition kernel runs at about
  1.2 TB/s on the H100; hopper's CART build went from 1.72 to 0.92 s with the
  hand-off below.
- **Staged upload:** with HMM (both GPU nodes) page-locking is slow, so the
  columns go through four page-locked 32 MB staging buffers instead: 6 GB in
  0.2 s instead of 0.35 s on the H100 node, the same 0.5 s on the A100 node.
- **Hand-off arena page-locked as needed:** CART with a fixed α hands only 2
  of airline's 12 GB to the CPU, so its arena is page-locked in 1 GB chunks
  ahead of the hand-offs instead of all at once (0.1-0.3 s). C4.5 keeps the
  single registration: chunks registered during the levels stall the GPU
  loop and made it 0.4 s slower.
- **`--gpu-min-rows` for CART:** rows / 8192, between 1024 and 2048 (airline:
  2048 instead of 4096). C4.5 keeps rows / 4096.
- **`chooseC45`** looks up `log2` in a table and allocates nothing.
- **`DT_PROFILE=1`** printed where C4.5 pruning spends its time (removed in
  November 2026: `--profile` records the same).

The parallel CPU build is unchanged (its code is the same; epyc1's C4.5 build
is 1% slower in every binary built after the pruning changes, which we put
down to code layout).

## October 2026: the node store and the GPU hand-off (mordor3 cluster)

Commit `0475249` (node store) and `f495aed` (GPU hand-off), measured on the
cluster's two GPU servers (CUDA) and two CPU servers (parallel), airline
(115M rows), one run per case. Every tree is byte-identical before and after.

- **Node store:** the builders take node ids from a shared store of
  fixed-size chunks. Threads took ids in blocks of 64 from the same chunk,
  so all of them reached the chunk's end together and waited while one
  allocated the next chunk under the lock. On hopper (2 sockets, `schedutil`
  governor) the waiting also let the clocks drop to about 0.9 GHz, and the
  CUDA backend's CPU threads ran half as fast on both sockets as on one.
  Now a thread takes a whole chunk (4096 nodes) and allocates it outside the
  lock, on its own socket.
- **GPU hand-off:** with the CPU subtrees faster, airline is fastest with
  nodes handed to the CPU at 2-4K rows instead of 28K. The automatic
  `--gpu-min-rows` (rows / 4096) is now capped at 4096; datasets up to
  16.8M rows keep their value.

| Airline | Node, backend | Before | Node store | + hand-off |
|---------|---------------|-------:|-----------:|-----------:|
| CART, unpruned | hopper cuda | 14.61 s | 4.98 s | 3.92 s |
| CART, unpruned | ampere cuda | 7.14 s | 5.29 s | 4.66 s |
| C4.5, unpruned | hopper cuda | 7.08 s | 4.82 s | 4.39 s |
| C4.5, unpruned | ampere cuda | 7.13 s | 6.02 s | 5.36 s |
| C4.5 | hopper cuda | 13.76 s | 11.38 s | 11.22 s |
| C4.5 | ampere cuda | 13.63 s | 12.25 s | 11.95 s |
| CART | ampere cuda | 3.34 s | 3.35 s | 2.84 s |
| CART, unpruned | dell parallel | 10.62 s | 10.25 s | – |
| C4.5 | dell parallel | 16.52 s | 16.04 s | – |

One socket is no longer faster anywhere: hopper's unpruned CART takes
6.13 s on one socket, 4.98 s on both. Other settings tried and left at their
defaults (no consistent gain): `--task-rows` 16K-1G, `--feature-parallel-rows`
8K and 128K on the CPU servers, `--gpu-min-rows` 1024 and 16384 on HIGGS.
(`--task-rows` and `--feature-parallel-rows` were removed in November 2026;
their defaults are fixed.)

## October 2026: tuning for many-core servers (mordor3 cluster)

Commit `a8563b3` (before) against `16cc93b` (after), measured on the cluster's
epyc1 node (2× EPYC 9255, 48 cores, `--parallel`) and hopper (H100 PCIe + 32
Xeon cores, `--cuda`). Before and after ran in the same job, interleaved, 3
runs each. Times are medians of `train total`. Every tree is byte-identical
before and after, on both backends.

| Dataset | Algorithm | Node, backend | Before | After | Speed-up |
|---------|-----------|---------------|-------:|------:|--------:|
| HIGGS (10.5M rows) | CART | epyc parallel | 1.95 s | 1.07 s | 1.8× |
| HIGGS | C4.5 | epyc parallel | 2.55 s | 1.86 s | 1.4× |
| airline (115M rows) | CART | epyc parallel | 15.98 s | 7.34 s | 2.2× |
| airline | C4.5 | epyc parallel | 24.67 s | 13.93 s | 1.8× |
| HIGGS | CART | hopper cuda | 1.48 s | 0.54 s | 2.8× |
| HIGGS | C4.5 | hopper cuda | 2.61 s | 1.65 s | 1.6× |
| airline | CART | hopper cuda | 11.64 s | 2.32 s | 5.0× |
| airline | C4.5 | hopper cuda | 26.03 s | 14.53 s | 1.8× |

What changed:
- **CART with a fixed α** does not grow nodes that pruning would cut back
  (docs/ALGORITHMS.md).
- **`--feature-parallel-rows` 4096 → 32768:** fewer thread wake-ups. A 48-core
  build had spent a third of its CPU time in the kernel.
- **Huge pages** for large blocks (`src/core/memory.cpp`).
- **`--gpu-min-rows`:** rows / 4096, at least 1024 (was 512).
- **GPU upload:** a page-locked dataset upload that works on the H100 node.
- **Parallel tree passes:** the copy, compaction and C4.5 bookkeeping run on
  all threads.


## October 2026: sweep, partition and pruning

Commit `d1cde64` (before) against `be8b236` (after), one run each, on the
desktop (i7-14700KF, 28 threads, DDR4-3200) and the laptop (i7-13650HX,
20 threads, RTX 5070 Laptop GPU). CART: `--cart -d 30 --alpha 1e-05`, C4.5:
`--c45`. Before and after are `train total` (presort, GPU setup, build,
pruning; loading and evaluation excluded); build, C4.5 pruning and peak
memory (after training) as before → after. Every tree is byte-identical: the dumps of each
dataset and algorithm have the same checksum on every backend and machine,
before and after.

| Dataset | Algorithm | Machine, backend | Before | After | Speed-up | Build (s) | Prune (s) | Peak (GB) |
|---------|-----------|------------------|-------:|------:|--------:|----------:|----------:|----------:|
| SUSY | CART | desktop serial | 9.48 s | 6.24 s | 1.52× | 8.95 → 5.76 | — | 1.1 → 1.1 |
| SUSY | C4.5 | desktop serial | 8.00 s | 6.23 s | 1.28× | 6.54 → 5.16 | 0.92 → 0.52 | 1.7 → 1.4 |
| SUSY | CART | desktop parallel | 1.33 s | 0.970 s | 1.37× | 1.09 → 0.73 | — | 1.8 → 1.7 |
| SUSY | C4.5 | desktop parallel | 1.77 s | 1.29 s | 1.37× | 1.18 → 0.87 | 0.31 → 0.16 | 2.3 → 2.0 |
| HIGGS | CART | desktop parallel | 4.89 s | 3.36 s | 1.46× | 4.05 → 2.59 | — | 6.1 → 5.9 |
| HIGGS | C4.5 | desktop parallel | 8.00 s | 5.69 s | 1.41× | 5.63 → 4.28 | 1.34 → 0.58 | 8.3 → 7.1 |
| SUSY | CART | laptop parallel | 1.30 s | 0.914 s | 1.42× | 1.13 → 0.77 | — | 1.7 → 1.7 |
| SUSY | C4.5 | laptop parallel | 1.39 s | 1.03 s | 1.35× | 0.96 → 0.74 | 0.23 → 0.14 | 2.3 → 2.0 |
| HIGGS | CART | laptop parallel | 4.63 s | 3.32 s | 1.39× | 4.09 → 2.81 | — | 5.4 → 5.9 |
| HIGGS | C4.5 | laptop parallel | 6.58 s | 5.19 s | 1.27× | 4.67 → 4.10 | 1.26 → 0.53 | 7.6 → 7.1 |
| SUSY | CART | laptop cuda | 0.620 s | 0.380 s | 1.63× | 0.52 → 0.31 | — | 1.3 → 1.3 |
| SUSY | C4.5 | laptop cuda | 0.865 s | 0.459 s | 1.88× | 0.51 → 0.23 | 0.23 → 0.14 | 1.9 → 1.6 |
| HIGGS | CART | laptop cuda | 2.00 s | 1.37 s | 1.46× | 1.66 → 1.18 | — | 3.8 → 3.8 |
| HIGGS | C4.5 | laptop cuda | 4.11 s | 1.97 s | 2.09× | 2.22 → 1.16 | 1.45 → 0.54 | 6.1 → 5.0 |

Where it comes from (details in [CPU.md](CPU.md) and [CUDA.md](CUDA.md)):

* CPU sweep: blocks of 16 cuts skipped by a bound on the gain over the
  block (two classes), so most cuts are only counted.
* CPU `--parallel`: big nodes partitioned out of place into a second copy of
  the columns, with the big children swept while their entries are written
  (each column read once per level instead of twice).
* C4.5 pruning: rows kept in leaf order in a row-major copy, routed 16 at a
  time over compact nodes; subtrees whose rows did not change are not pruned
  again after raising.
* Cuda: a single-pass partition that also counts the next level's tile
  histograms; the exact (double precision) pass scores only the few cuts the
  float pass kept per tile; a Gini estimate with a relative error bound (the
  float pass was removed in November 2026: see above);
  C4.5's sorted values copied into memory page-locked during setup.

HIGGS CART on the laptop GPU is limited by the CPU threads finishing the small
subtrees (the `wait cpu subtrees` phase of `--profile` shows it), so part of the GPU's gain does not
show in its total. The second column copy of `--parallel` costs memory at
low thread counts (HIGGS, 4 threads: +1.9 GB); at 20–28 threads it mostly
replaces the presort's per-thread buffers (peak −1.2 to +0.5 GB above).
