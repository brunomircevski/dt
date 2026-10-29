# ECBDL'14 (631 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** `ecbdl`: all 31,992,921 training rows, on the CPU nodes only (the GPUs would need about 160 GB); `ecbdl_15m`: the first 15,000,000, on every node.
- **Test rows:** the competition's labelled test set, 2,897,917 rows. Until 2026-10-23 100,000 random training rows were the test set: cluster-ecbdl-old (removed; last in f4668df).
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning. The pruned runs of 2026-10-23 (CART α = 1e-5, C4.5 CF 0.25) are in git at 96705a7.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB); parallel on every CPU of both sockets on epyc1 and dell1 (GPUs on `ecbdl_15m` only); --serial and --simple (one thread) on epyc. Three runs per case, averaged, except --simple: one run.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart-ecbdl.png` (full) and `chart-ecbdl_15m.png`, `bench/testset_chart.py` (log scale).
- **Reruns queued (2026-10-28, jobs 381–545, `bench/cluster/submit-variants-2026-10-28.sh` and `submit-topup-2026-10-28.sh`):** the missing variants, and more runs: a case whose run takes about 2 minutes or less (the whole process) gets 10 runs, one under about 10 minutes 5, a longer one keeps what it has. The runs here count; only the missing ones run (into `cluster-ecbdl-new/` and `-top/` on mordor3), and `bench/merge_runs.py` adds them to the run folders here when done. `ecbdl_15m`: CUDA hybrid and GPU only on hopper and ampere, parallel on both sockets and on one socket (new) and --serial on epyc, 10 runs each (--serial: 5, 6–7 minutes a run). `ecbdl`: parallel on both sockets and on one socket (new) on epyc, 5 runs each (about 4 minutes a run with reading the data; --serial, 15–18 minutes, keeps its three), and one run of each GPU backend, which needs about 160 GB of VRAM, to record how it fails. --simple (over 10 minutes) keeps its one run. The chart draws mean ± standard deviation and the number of runs, and marks the folders still waiting as "queued" or "still running".
- **Spin stall, fixed 2026-10-28 (rerun queued, jobs 550–570, `bench/cluster/submit-spinfix-2026-10-28.sh`):** the parallel backend partitions a big node (≥ 1M rows, few columns to move) in blocks taken in order by all threads, each block waiting for the one before it. It waited by spinning (`_mm_pause`) without end: when the thread of the block before was descheduled, the other threads spun and kept it off a CPU, so some runs took 2–8× as long (epyc chess_10m: user time 5 s instead of 2 s, the same system time; 11 of 80 runs > 2× the median). It now spins 2,048 pauses, then sleeps (`atomic::wait`, woken by the block before). On epyc2, 40 runs each of chess_10m (48 threads): CART 0.155 ± 0.131 s → 0.087 ± 0.005 s, C4.5 0.186 ± 0.163 s → 0.095 ± 0.005 s, no run above 2× the median. Only the parallel backend has those blocks (CUDA hybrid gives the CPU nodes under 4,096 rows; --serial and --simple have no pool), so only the parallel epyc runs (both sockets, one socket) start over, with 10 runs a case (full ECBDL 5), into `cluster-<dataset>-fix/`, replacing the old folders when done. The dell runs of ./tree_cpu run again too (below); until the reruns are in, the bars of old builds are marked (hatched, ⚠, from `problems.json`), as are runs more than twice the mean of the other runs.
- **One build for every epyc ./tree_cpu run (2026-10-28, `bench/cluster/submit-final-2026-10-28.sh`):** after the spin fix, C4.5's list of distinct values (its thresholds are training values) was made allocation-free: counted, then written in place on the pool, and in one pass with --serial (SOREL's C4.5 presort on epyc varied 11–34 s; see cluster-sorel). Same trees as before in every check (SOREL, HIGGS, airline, ECBDL 15M, serial and parallel). So that every epyc run of ./tree_cpu is of that one build, the parallel runs started before it (jobs 550–556) run again, and --serial runs again for every dataset (10 runs a case of about 2 minutes or less, 5 under about 10 minutes, longer ones once), into `cluster-<dataset>-fix/`, replacing the old folders. The long runs (full ECBDL and SOREL parallel, every --serial, then --simple) run one after another on epyc2, the short parallel ones on epyc1. CUDA hybrid, GPU only and --simple do not use that code and are not rerun. Every dell run of ./tree_cpu (parallel on both sockets, on one socket, dell2's 512 GB, --serial) runs again on the same build too (`bench/cluster/submit-dell-chess-2026-10-28.sh`, jobs 592–618: parallel on dell1, then --serial there; dell2 only its 512 GB runs), the same runs as before, replacing the old folders; --simple on dell stays.
- **Every run again on one build each, in batches (2026-10-28 evening, `bench/cluster/submit-batches-2026-10-28.sh`, jobs 636–645):** the pending jobs above were cancelled and replaced by a few batch jobs (`bench/cluster/batch.sh`: one SLURM job per node and group, its runs one after another). CUDA hybrid and GPU only on hopper and ampere run again on the new ./tree (67495f9, the CUDA rework: the smaller side's goes-left bitmap, the split column by position, upload overlapped with the presort, root counts on the GPU), installed by job 636 only after its trees matched the old build's (HIGGS, airline, ECBDL 15M; CART and C4.5, both GPU backends; 10–25 % faster there); ECBDL 15M only, the full set does not fit in VRAM. On the CPU, dell and epyc for every backend (--simple, --serial, parallel on one socket, on both), on the final ./tree_cpu; the runs already done on it are kept, as are --simple's (that code did not change). Runs per case: about 2 minutes or less 10, under about 10 minutes 3, longer once; long runs on epyc2 (epyc1 stays free for the short ones), every dell run on dell1, short batch first. Test accuracy is now predicted once per distinct tree (`bench/run.py`), not once per run. The chart groups the bars by backend, the nodes under each; it reads `cluster-ecbdl-fix/` too: a rerun done replaces the old bar, one pending marks it ("old build; rerun queued") or shows as "queued".

## Results

Run at 74a011f: the GPU and parallel runs on 2026-10-23. Run at 8187eb9: --serial and --simple on epyc on 2026-10-27; `simple-epyc-ecbdl_15m` holds the --simple runs of cluster-cpu-vs-others (`vs-epyc-ecbdl_15m-simple-*`, the same command). The tables give means of ./tree's training time. Accuracies, nodes and depth are the same in every run.

![ECBDL'14, all training rows](chart-ecbdl.png)

![ECBDL'14, first 15M training rows](chart-ecbdl_15m.png)

- **Same tree everywhere:** on each training set, every backend grew the same tree in every case.
- **Accuracy is not informative:** only about 1.7% of the test rows are contacts, so the majority class alone scores 98.32%. The trees match or barely miss that score. A class-weighted or AUC measure would be needed to compare them.
- **15M sample:** the H100 takes 3.7–3.9 s, 1.8–2.1× faster than epyc (6.8–8.2 s); the A100 takes 5.8–6.0 s. GPU only needs 36 GiB of RAM, against 107 GiB for hybrid; epyc and dell use 111 GiB.
- **Full set (CPU only):** epyc takes 14.5–18.5 s, dell 22–28 s, with 238 GiB of RAM. That is 2.1–2.2× the 15M sample's time for 2.1× the rows.
- **Serial and simple:** --serial takes 35–38× the parallel run (544 s and 699 s on the full set). --simple takes 5,857 s and 6,890 s on the full set, 10–11× --serial, with a third of the RAM (76 GiB).

### ecbdl

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 98.30% | 97.86% | 1,917 | 10 |
| C4.5, no pruning | 98.32% | 97.86% | 785 | 10 |

| Run | CART | C4.5 | RAM GiB (peak) |
|---|---|---|---|
| epyc, parallel, both sockets | 14.5 s | 18.5 s | 237.6–237.7 |
| dell, parallel, both sockets | 22 s | 27.8 s | 237.6–237.7 |
| epyc, --serial | 544 s | 699 s | 226.4 |
| epyc, --simple | 5,857 s | 6,890 s | 75.7–75.9 |

### ecbdl_15m

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 98.31% | 97.92% | 1,863 | 10 |
| C4.5, no pruning | 98.32% | 97.92% | 725 | 10 |

| Run | CART | C4.5 | RAM GiB (peak) | VRAM GiB (peak) |
|---|---|---|---|---|
| hopper, CUDA hybrid | 3.74 s | 3.85 s | 106.5 | 73.3 |
| hopper, GPU only | 3.73 s | 3.84 s | 36.0 | 73.3 |
| ampere, CUDA hybrid | 5.84 s | 6.01 s | 106.5 | 73.3 |
| ampere, GPU only | 5.81 s | 5.95 s | 36.0 | 73.3 |
| epyc, parallel, both sockets | 6.79 s | 8.23 s | 111.4 | – |
| dell, parallel, both sockets | 10.5 s | 12.3 s | 111.4 | – |
| epyc, --serial | 240 s | 302 s | 106.2 | – |
| epyc, --simple | 2,273 s | 2,587 s | 35.5–35.6 | – |
