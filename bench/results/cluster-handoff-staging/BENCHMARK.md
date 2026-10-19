# CUDA hand-offs without page-locking the arena (mordor3, 2026-10-19)

The CUDA grower hands small nodes to the CPU threads by copying their columns
into a host buffer, the arena (room for every column of every row: 75 GB for
ECBDL'14's 15M sample, 12 GB for airline). Until e0a9106 the arena was
page-locked (`cudaHostRegister`) in a background thread, so the copies could
write into it directly. The registration holds the CUDA driver's lock, and the
GPU loop waited for it: 1.2–1.9 s for ECBDL (see
[cluster-ecbdl-profile](../cluster-ecbdl-profile/BENCHMARK.md)), and in these
jobs up to 3.4 s on hopper and 4.8–11.5 s on ampere, depending on the node's
memory state.

**The change:** the arena stays pageable.
- Each hand-off piece (at most a 128 MB part of the device ring) is downloaded
  into one of two page-locked 128 MB buffers, one per copy stream.
- A host function on that stream copies the buffer into the arena on the
  staging threads (the GPU's NUMA node).
- The two streams take turns, so one downloads while the other's buffer is
  copied out.
- The arena is still touched on all threads during the presort, in chunks and
  in order. A hand-off waits only for the part it fills; CART with a fixed α,
  which hands over little, skips the touch.
- C4.5's arranged rows (pruning) are downloaded through the staging buffers too.

**Same trees everywhere** (md5 of `--dump` in every run below). compute-sanitizer
memcheck: 0 errors (SUSY 500K, C4.5 and unpruned CART, both GPUs).

## Result (final version, `logs/ab5-*.log`)

Base = e0a9106, staged = the change. Both alternate in one job per node
(`--exclusive`, all cores, `--profile` on). Median train total (runs in
brackets): 3 runs for airline and HIGGS, 2 for ECBDL.

| node | case | base (s) | staged (s) | change |
|---|---|--:|--:|--:|
| hopper | ECBDL 15M C4.5 | 52.73 (52.64 52.81) | 51.25 (51.03 51.47) | −2.8% |
| hopper | ECBDL 15M CART, no pruning | 18.73 (18.73 18.73) | 17.60 (17.56 17.65) | −6.0% |
| hopper | ECBDL 15M CART α 1e-5 | 8.15 (8.12 8.19) | 8.06 (8.04 8.08) | −1.2% |
| hopper | airline C4.5 | 6.19 (6.19 5.91 8.64) | 6.08 (6.08 5.20 6.38) | −1.8% |
| hopper | airline CART, no pruning | 4.19 (4.19 5.33 3.92) | 3.31 (4.72 3.31 3.17) | −21.0% |
| hopper | airline CART α 1e-5 | 1.10 (1.10 1.19 1.07) | 1.10 (1.10 1.13 1.08) | 0.0% |
| hopper | HIGGS C4.5 | 1.10 (1.27 1.09 1.10) | 1.24 (1.24 1.27 1.23) | +12.7% |
| hopper | HIGGS CART, no pruning | 0.84 (0.88 0.84 0.80) | 0.85 (0.88 0.85 0.80) | +1.2% |
| hopper | HIGGS CART α 1e-5 | 0.43 (0.47 0.43 0.43) | 0.41 (0.41 0.39 0.41) | −4.7% |
| ampere | ECBDL 15M C4.5 | 74.68 (72.76 76.59) | 65.73 (65.90 65.56) | −12.0% |
| ampere | ECBDL 15M CART, no pruning | 31.73 (31.62 31.83) | 23.89 (23.82 23.97) | −24.7% |
| ampere | ECBDL 15M CART α 1e-5 | 11.78 (11.82 11.74) | 11.66 (11.67 11.64) | −1.1% |
| ampere | airline C4.5 | 5.95 (5.96 5.93 5.95) | 5.65 (5.70 5.65 5.63) | −5.0% |
| ampere | airline CART, no pruning | 3.79 (3.81 3.78 3.79) | 3.65 (3.65 3.65 3.70) | −3.7% |
| ampere | airline CART α 1e-5 | 1.62 (1.65 1.62 1.62) | 1.63 (1.62 1.63 1.63) | +0.6% |
| ampere | HIGGS C4.5 | 1.18 (1.15 1.18 1.23) | 1.12 (1.11 1.13 1.12) | −5.1% |
| ampere | HIGGS CART, no pruning | 0.96 (0.96 0.89 0.99) | 0.89 (0.89 0.89 0.90) | −7.3% |
| ampere | HIGGS CART α 1e-5 | 0.53 (0.53 0.54 0.52) | 0.52 (0.50 0.52 0.53) | −1.9% |

- **The stall is gone.** In base, the registration lands in grow setup,
  level 0 or a later level, depending on when the arena's touch finishes. In
  this job ampere's grow setup took 6.1–11.5 s for ECBDL; hopper's level loop
  was 1.2–1.4 s longer (NUMA node 0 had 11 GB free of 516). Staged, grow setup
  plus level 0 take at most 0.22 s in every case, whatever the node's memory
  state.
- **ECBDL:** the hand-offs (30–45 GB) are spread over 91–272 levels, and the
  staging keeps up with them: the level loop is as long as base's without the
  stall (ampere C4.5: 33.9–34.0 s both), or shorter by the stall (hopper; ampere
  unpruned CART 23.0 → 21.2–21.4 s).
- **Airline:** its hand-offs (11.6 GB) come in a burst of a few levels. On
  hopper the levels then take longer than with the registered arena (ab4:
  1.34–1.46 s against 1.18 s), because the copy into the arena is slower than
  the PCIe link (55 GB/s). The registration it saves (0.2 s, 1.4–1.6 s in ab4's
  hopper job) balances that out.
- **HIGGS C4.5 on hopper is 0.1 s slower:** the pruning's arranged rows
  (1.2 GB) now go through the staging buffers (0.05 → 0.08 s), and its subtrees
  took 0.36–0.39 s instead of 0.30–0.36 s. Over ab4 and ab5 the medians are
  1.24 s (staged) and 1.18 s (base).

## Variants tried on the way

| log | variant | outcome |
|---|---|---|
| `ab-*.log` | 4 × 32 MB buffers, no touch of the arena | ECBDL as above. Hopper's airline C4.5 5.11 → 5.2–6.1 s: its levels 1.18 → 1.8–2.0 s, because the copies faulted in the arena's pages. |
| `ab2-*.log` | + `MADV_POPULATE_WRITE` of the arena in the background | Hopper's airline levels 1.6–1.8 s. ECBDL fine with 93 GB free on node 0. |
| `ab3-*.log` | buffer size 32 or 128 MB; copies on the copier threads or the main pool | 128 MB: levels 0.2–0.4 s shorter on both GPUs. The main pool gained nothing. |
| `ab4-*.log` | 2 × 128 MB, one per stream, + populate | With 28 GB free on hopper's node 0, the populate (it holds the process's mmap lock) stretched the presort from 0.5 to 4.1–4.4 s: ECBDL +1.3–3.5 s. Hence the memset touch of the final version. |
| `ab5-*.log` | final | the table above |

## Reproduce

Two checkouts side by side (`base/`, `new5/`, each built with
`bench/cluster/build.sh` on hopper), then from their parent:

```
sbatch -p hopper --exclusive scripts/ab.sh
sbatch -p ampere --exclusive scripts/ab.sh
python3 scripts/summary.py ab5-hopper.log ab5-ampere.log
```
