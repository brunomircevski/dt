# The four dell nodes

Checked on 2026-11-17 (jobs 331–334, each node `--exclusive`): `lscpu`, memory per
NUMA node, governor, kernel, and `bench/cluster/triad.c` (`gcc -O3 -fopenmp`; copy and
triad over 3 × 4 GiB arrays, best of 10; three times with 48 threads,
`OMP_PROC_BIND=spread`, then once with one thread).

| Node | RAM | Triad, 48 threads | Triad, 1 thread |
|---|---|--:|--:|
| dell1 | 1 TB (16 × 64 GB, every channel) | 379–380 GB/s | 18.0 GB/s |
| dell2 | 512 GB (8 × 64 GB) | 210–212 GB/s | 17.4 GB/s |
| dell3 | 512 GB (8 × 64 GB) | 212 GB/s | 17.2 GB/s |
| dell4 | 512 GB (8 × 64 GB) | 211–212 GB/s | 17.4 GB/s |

All four: 2× Xeon Gold 6542Y, 4 NUMA nodes (the same CPU numbering), `powersave`
(intel_pstate), kernel 7.0.0-42. dell2–4 are interchangeable; dell1 has 1.8× their
bandwidth with 48 threads and the same with one. The benchmarks use dell2–4 only
(the paper's dell is a 512 GB node); runs on dell1 count as out of date
(`benchlib.RETIRED_HOSTS`).

Earlier measurements (docs/PERFORMANCE.md, "Nodes"): `--multicore` on dell2–4 took
1.13–1.46× as long as on dell1 (SUSY 1.13×, MNIST8M up to 1.46×); one-thread runs
0.9–1.2% longer (2026-11-09/10, SUSY and HIGGS).
