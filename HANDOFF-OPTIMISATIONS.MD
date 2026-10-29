# CART / C4.5 performance work (2026-10-28/29): finished

The brief was to make the C++ CART / C4.5 builder faster or leaner without changing a single tree. Experiments ran unpruned (`--no-prune -d 10`) on mordor3 compute nodes, CPU on dell (48 threads) and CUDA on hopper (H100) / ampere (A100). Every run's `--dump` was compared with the base binary's.

The work is done. The experiment branches (`exp/perf-d10`, `merge/perf-d10`, and the older `wip/arena-pagelock-fix`) are deleted, as is the server clone `~/dt-perf`. This file is the record of the results.

## Merged into main

**CPU presort (commit "CPU presort: column histograms in chunks on all threads; radix over the varying key bits only", `src/build/cpu_builder.cpp` only):**
- **HISTPAR:** the presort digit histograms of big columns are counted in chunks on all threads, not one thread per column.
- **RADIX:** the radix sort runs only over the key bits that vary. Integer columns are keyed by `int(v) − min`, with 12-bit digits, for columns of ≥ 1M rows only. The old `radixHistogram()` is gone.

Check against main 8bdf641's code (jobs 775/776, dell2 CART, dell3 C4.5, median of 3 interleaved reps). `tests/check.sh` passes and all 24 trees are identical:

| dataset | presort CART | train CART | presort C4.5 | train C4.5 |
|---|---|---|---|---|
| chess_1b par | −62% | **−32%** | −49% | **−20%** |
| chess_100m par | −52% | −24% | −49% | −19% |
| chess_10m par | −37% | −17% | −42% | −6% |
| airline par | −34% | −7% | −29% | −2% |
| ecbdl_15m par | −25% | −4% | −19% | −6% |
| susy par | −17% | −0% | −15% | −12% |
| covertype par | −9% | −4% | −7% | −7% |
| mnist8m par | −3% | −2% | −8% | +3% (build noise) |
| higgs par | +3% | +1% | +3% | −2% |
| covertype / susy / higgs serial | −2 / −9 / −4% | −0 / −2 / −1% | −2 / −5 / −3% | −0 / −2 / −0% |

Build time is unchanged within the dell noise (±8%; chess C4.5 up to ±18%). The +3% higgs presort is noise: its 28 columns of 11M rows keep all threads busy without chunks, and RADIX saves no pass on its float columns.

## Not merged

| Change | Result |
|---|---|
| **NOWRITE** (CPU): fused child sweeps at depth max−1 skip writing columns that are never read again | build −5% to +3%, noise |
| **PREFETCH** (CPU): partition loops prefetch the left/right flag 32 rows ahead | within ±3% |
| **PACK** (CUDA): upload integer columns as uint8/16 and unpack them in a kernel | trees identical, but slower everywhere (below) |
| **SORTBITS** (CUDA): cub radix over the varying key bits only | trees identical, but slower everywhere (below) |
| **Chunked arena page-locking** (`wip/arena-pagelock-fix`, 2026-10-18) | wins on hopper ECBDL, but C4.5 is +3% on ampere and airline C4.5 crashes in chunk mode. Its count-table part (no zero-fill) was already on main. |

**CUDA timing (jobs 721 hopper / 722 ampere, 2026-10-29):** all 76 tree dumps are identical. Median train time, 3 reps, compared with the experiment binary with its switches off. The off binary is as fast as base.

| dataset | hopper PACK / SORTBITS | ampere PACK / SORTBITS |
|---|---|---|
| covertype CART | +28% / +18% | +9% / +3% |
| susy CART | +52% / +53% | +24% / +19% |
| higgs CART | +48% / +103% | +40% / +30% |
| chess_100m CART | +55% / +51% | +34% / +23% |
| airline CART | +105% / +86% | +35% / +33% |
| mnist8m CART | +56% / +40% | +14% / +19% |
| ecbdl_15m CART | +38% / +36% | +22% / +30% |
| chess_1b CART | +77% / +85% | +28% / +26% |
| higgs C4.5 | +92% / +40% | +26% / +28% |
| airline C4.5 | +97% / +84% | +25% / +29% |
| ecbdl_15m C4.5 | +41% / +35% | +20% / +26% |
| chess_1b C4.5 | +53% / +77% | +26% / +26% |

All of the extra time is in setup (upload + presort); build time is unchanged. Both switches go through a new upload path (`uploadPacked`) that scans each staged chunk on the host before it is sent. That costs more than PACK saves in bytes or SORTBITS saves in radix passes: airline CART user time goes from 33 s to 58 s on hopper. Any future try needs the scan hidden in the existing copy pipeline, for example fused into the staging memcpy, rather than as an extra pass.

## Parallel hang (fixed on main in 8bdf641)

About 1 in 40 big `--parallel` runs on dell hung forever. The cause is a lost wake-up in libstdc++ 13's `atomic::notify` after a `memory_order_release` store. `notify_*` makes the futex wake call only if it sees a waiter in the bucket's waiter count. That load can run before the store is visible (on x86 the store is a plain `mov`), so the waiter sleeps on the old value for good. It was caught under gdb in `partitionInBlocks` (`lefts[]`) and in the column partition (`done[]`). Main's `waitNonZero()` avoids `atomic::wait` altogether: it spins, then yields, then sleeps. A `seq_cst` store also fixed it (160 runs, 0 hangs).
