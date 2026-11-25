# The 2026-11-24 hand-off re-sweep: depths 15–30, and no limit again

**Why.** Until now `handoff_rows.json` held only `depth10` and `nodepth`, and
`run.py` gave depths 15–30 the `nodepth` value (the shallowest stored limit at
least as deep as the run). Those depths had never been swept, and the
`nodepth` sweep (2026-11-15/16, `bench/hybrid-vs-gpu-only/sweep/`) predates
./tree 7f98876f, the build of the final GPU results. So hybrid-vs-gpu-only's
deep points did not run at the best value.

**What.** Only what a time benchmark runs: hybrid-vs-gpu-only's datasets
(HIGGS, MNIST8M, SOREL 4M) on both GPU nodes, at depth 15, 20, 25, 30
(`depth<N>/`), and the `nodepth` value checked again around its pick
(`nodepth/`: the pick and two neighbours each side by factors of 2, and
./tree's default where it is another value). `depth10` stays: its sweeps are
flat (within 1–2%). `handoff_sweep.sh`, 5 runs of CART and C4.5 (unpruned)
per value, scored by `pick_handoff.py` (training time without the GPU setup;
the hand-off buffers have a fixed size, so the setup does not depend on the
value). Values: HIGGS default, 128–8192; MNIST8M default, 512–32,768;
SOREL 4M default, 4096–65,536 (its curve falls steadily up to 16,384,
`bench/hybrid-vs-gpu-only/BENCHMARK.md`).

**When.** ./tree 0b6ff509 (`src/` as 7f98876f), clone `~/dt-r24` built by job
894; 2026-11-24 17:18 – 2026-11-25 05:11, `--exclusive`: hopper jobs
902, 903, 904, 909; ampere 905, 906, 907, 910 (`logs-2026-11-24/`, wrapper
`sweep.sh`: A HIGGS + MNIST8M depths, B SOREL 4M depths, C no limit, D the
values a job missed). Both nodes created `depth15/` at the same moment and
NFS refused ampere's HIGGS (`mkdir: File exists`); job 910 measured it.

## The picks

SOREL 4M's depths 15–30 were swept again below 4096 on 2026-11-25 (below):
depth 15 now takes 2048 on both nodes; the other picks stand.

Each value's score is its mean over CART and C4.5 of the time over that
protocol's fastest; the columns give how much slower than the pick the
runner-up, the value the results had used, and ./tree's default scored.

| Node | Dataset | Depth | Value used | New pick | Runner-up | Used value | ./tree default |
|---|---|---|---|---|---|---|---|
| hopper | HIGGS | 15 | 256 | 512 | 1024 (+1.4%) | +3.1% | +0.3% |
| hopper | HIGGS | 20 | 256 | 1024 | 512 (+2.9%) | +10.7% | +3.8% |
| hopper | HIGGS | 25 | 256 | 1024 | 512 (+0.4%) | +2.4% | −0.9% |
| hopper | HIGGS | 30 | 256 | 512 | 1024 (+1.6%) | +3.4% | +1.2% |
| hopper | HIGGS | none | 256 | 512 | 1024 (+1.9%) | +2.9% | – |
| hopper | MNIST8M | 15 | 8192 | 2048 | 4096 (+0.5%) | +5.1% | +4.4% |
| hopper | MNIST8M | 20 | 8192 | 4096 | 2048 (+1.1%) | +7.3% | +6.1% |
| hopper | MNIST8M | 25 | 8192 | 4096 | 2048 (+2.8%) | +10.1% | +8.9% |
| hopper | MNIST8M | 30 | 8192 | 4096 | 2048 (+4.9%) | +9.9% | +8.9% |
| hopper | MNIST8M | none | 8192 | **same** | 4096 (+5.0%) | – | −0.5% |
| hopper | SOREL 4M | 15 | 16384 | 4096 | 8192 (+0.8%) | +3.8% | +3.4% |
| hopper | SOREL 4M | 20 | 16384 | 4096 | 8192 (+0.1%) | +3.6% | +3.1% |
| hopper | SOREL 4M | 25 | 16384 | 8192 | 4096 (+0.2%) | +3.6% | +3.6% |
| hopper | SOREL 4M | 30 | 16384 | 8192 | 4096 (+0.2%) | +1.6% | +1.4% |
| hopper | SOREL 4M | none | 16384 | **same** | 8192 (+1.9%) | – | – |
| ampere | HIGGS | 15 | 512 | 1024 | 4096 (+0.4%) | +1.9% | +2.3% |
| ampere | HIGGS | 20 | 512 | 2048 | 1024 (+4.2%) | +20.8% | +19.4% |
| ampere | HIGGS | 25 | 512 | 1024 | 2048 (+1.5%) | +7.7% | +6.9% |
| ampere | HIGGS | 30 | 512 | 1024 | 512 (+3.5%) | +3.5% | +4.1% |
| ampere | HIGGS | none | 512 | 1024 | 512 (+3.2%) | +3.2% | – |
| ampere | MNIST8M | 15 | 8192 | 2048 | 4096 (+0.2%) | +3.8% | +2.8% |
| ampere | MNIST8M | 20 | 8192 | 4096 | 2048 (+1.4%) | +3.5% | +3.6% |
| ampere | MNIST8M | 25 | 8192 | 4096 | 2048 (+4.2%) | +5.9% | +5.0% |
| ampere | MNIST8M | 30 | 8192 | 4096 | 8192 (+4.9%) | +4.9% | +4.0% |
| ampere | MNIST8M | none | 8192 | **same** | 4096 (+4.6%) | – | −0.1% |
| ampere | SOREL 4M | 15 | 16384 | 4096 | 8192 (+0.9%) | +3.6% | +3.6% |
| ampere | SOREL 4M | 20 | 16384 | 4096 | 8192 (+0.4%) | +3.7% | +4.8% |
| ampere | SOREL 4M | 25 | 16384 | 4096 | 8192 (+0.3%) | +8.8% | +8.2% |
| ampere | SOREL 4M | 30 | 16384 | 4096 | 8192 (+0.5%) | +15.4% | +15.4% |
| ampere | SOREL 4M | none | 16384 | **same** | 8192 (+1.3%) | – | – |

- Every depth 15–30 pick differs from the `nodepth` value they used:
  smaller (HIGGS 512–2048, MNIST8M 2048–4096, SOREL 4M 4096–8192), and the
  used value scored 1.6–20.8% slower. ./tree's default
  (`max(512, min(10 × features, 2^depth, 16384))`) is close to the used value
  for MNIST8M and SOREL 4M and as slow.
- No limit: MNIST8M (8192) and SOREL 4M (16,384) keep their pick; HIGGS moves
  from 256 to 512 on hopper and from 512 to 1024 on ampere (the old values 2.9%
  and 3.2% slower).
- Close calls: many runners-up are within 2% of the pick. They were not
  measured again: either value is within the sweep's noise, and no close call
  decides whether a result had to be measured again.
- HIGGS depth 30 on hopper picks 512, ./tree's default: cluster-gpu-vs-others'
  depth-30 line (`HANDOFF=default`) already ran at the best value.

## Measured again

Every hybrid-vs-gpu-only `--hybrid` point whose value changed, 10 runs a case,
GPU only and the depth 5/10 points as they were: depths 15–30 of the three
datasets and HIGGS without a limit, on both nodes, and the MNIST8M profile at
depths 15–30 (`bench/hybrid-vs-gpu-only/BENCHMARK.md`). Not measured again:
MNIST8M and SOREL 4M without a limit, cluster-power (MNIST8M no limit),
cluster-gpu-vs-others, everything at depth 10 and shallower.

Runs: 2026-11-25 08:39–12:08, hopper job 939, ampere job 940
(`logs-2026-11-24/rerun.sh`: `hybrid_vs_gpu_only.sh` with `KINDS=cuda RUNS=10`
into fresh folders, then `merge_runs.py --replace`). Every tree is the same as
before (node counts). The new values speed up CART most (hybrid time, old →
new): MNIST8M depth 30 6.47 → 5.54 s on hopper, 7.84 → 7.20 s on ampere;
SOREL 4M depth 30 6.62 → 6.35 s and 11.2 → 9.03 s; HIGGS depth 20 0.587 →
0.555 s and 0.668 → 0.590 s. C4.5 moves within ±2% (the pick minimises the
mean over both). Some single cases got slower by up to 3% (HIGGS CART depth 25
on hopper, HIGGS CART without a limit on ampere), the C4.5 run of the same
value faster. The GPU-only / hybrid ratios that changed most: SOREL 4M CART
depth 30 on ampere 0.95 → 1.18×, MNIST8M CART depth 30 on hopper 1.32 → 1.54×,
HIGGS CART depth 20 on ampere 0.89 → 1.01×.

## SOREL 4M below 4096 (2026-11-25)

The sweep above started SOREL 4M at 4096, and 4096 came out fastest in 6 of
its 8 depth-limited cases, so a smaller value could have been faster still
(its depth-10 pick is 1024). Swept again on 2026-11-25, depths 15–30 on both
nodes: 1024, 2048, 4096 and 8192, 4096 and 8192 measured again in the same job
(the 2026-11-24 runs of those two moved to `depth<N>/old-2026-11-24/`; 16,384
and up and `default` are the 2026-11-24 runs). ./tree cfd18321 (`src/` as
7f98876f), clone `~/dt-r25` built by job 951, `--exclusive`: hopper job 952,
ampere job 953, 15:35–18:40 (`logs-2026-11-25/`).

| Node | Depth | 1024 | 2048 | 4096 | 8192 | Pick |
|---|---|---|---|---|---|---|
| hopper | 15 | +0.5% | **pick** | +0.6% | +1.3% | 2048 (was 4096) |
| hopper | 20 | +1.7% | +0.4% | **pick** | +0.1% | 4096 |
| hopper | 25 | +3.2% | +1.1% | +0.4% | **pick** | 8192 |
| hopper | 30 | +3.7% | +1.5% | +0.2% | **pick** | 8192 |
| ampere | 15 | +0.3% | **pick** | +0.1% | +1.0% | 2048 (was 4096) |
| ampere | 20 | +1.4% | +0.4% | **pick** | +0.3% | 4096 |
| ampere | 25 | +2.8% | +1.1% | **pick** | +0.3% | 4096 |
| ampere | 30 | +3.6% | +1.7% | **pick** | +0.5% | 4096 |

Below 4096 the curve rises again from depth 20 on, so those picks were the
fastest after all; only depth 15 moves, to 2048, by 0.1–0.6% (within the
sweep's noise, but the pick is the pick).

Measured again: hybrid-vs-gpu-only's SOREL 4M `--hybrid` at depth 15, 10 runs
of CART and C4.5 a node (`logs-2026-11-25/rerun.sh`, hopper job 968, ampere
967, 2026-11-25 19:03–19:40, merged with `merge_runs.py --replace`). The same
trees; hybrid time, old → new: hopper CART 3.99 → 3.95 s, C4.5 4.27 → 4.25 s;
ampere CART 5.75 → 5.71 s, C4.5 6.02 → 6.06 s (GPU-only / hybrid: hopper 1.03
→ 1.04× and 0.99×, ampere 1.01 → 1.02× and 0.98×).

## Also on 2026-11-25

- **cluster-power, ECBDL'14 on hopper:** its Hybrid had run ./tree's default
  (1024 at depth 10) instead of the stored 4096; job 954 measured it again at
  4096 (2.97 → 2.92 s CART, 2.95 → 2.93 s C4.5, the energy unchanged at 2.09
  and 2.10 kJ; `bench/cluster-power/BENCHMARK.md`).
- **./tree's default `--handoff-rows`** is now 10 × features between 1024 and
  16,384, with a depth limit D at most 2^D and 4096 (`docs/PERFORMANCE.md`),
  fitted to these sweeps: 0.3% slower than the fastest on average and 4.2% at
  most, against 3.6% and 20.8% for the earlier
  `max(512, min(10 × features, 2^D, 16384))`. The `./tree default` column
  above is the earlier rule. No benchmark result depends on the default: chess
  runs it, but hands nothing off at depth 10, and cluster-gpu-vs-others' HIGGS depth-30 line ran `HANDOFF=default` (512 then,
  also its sweep value) and now takes the stored 512 (`final.sh`).
