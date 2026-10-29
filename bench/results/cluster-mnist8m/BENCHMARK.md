# MNIST8M (784 features, 10 classes): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** infinite MNIST's 8,100,000 distorted digits (LIBSVM `mnist8m`), 784 pixel features (0–255).
- **Test rows:** the original MNIST test set, 10,000 rows (LIBSVM `mnist.t`).
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB); on epyc1 and dell1 parallel on both sockets, parallel on one socket, --serial and --simple. Three runs per case, averaged, except --simple: one run.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart.png`, `bench/testset_chart.py` (log scale).
- **Reruns queued (2026-10-28, jobs 381–545, `bench/cluster/submit-variants-2026-10-28.sh` and `submit-topup-2026-10-28.sh`):** the missing variants, and more runs: a case whose run takes about 2 minutes or less (the whole process) gets 10 runs, one under about 10 minutes 5, a longer one keeps what it has. The runs here count; only the missing ones run (into `cluster-mnist8m-new/` and `-top/` on mordor3), and `bench/merge_runs.py` adds them to the run folders here when done. CUDA hybrid and GPU only on hopper and ampere, parallel on both sockets and on one socket and --serial on epyc, 10 runs each (--serial: 5, 5 minutes a run); --simple (over 10 minutes) keeps its one run. The chart draws mean ± standard deviation and the number of runs, and marks the folders still waiting as "queued" or "still running".
- **Spin stall, fixed 2026-10-28 (rerun queued, jobs 550–570, `bench/cluster/submit-spinfix-2026-10-28.sh`):** the parallel backend partitions a big node (≥ 1M rows, few columns to move) in blocks taken in order by all threads, each block waiting for the one before it. It waited by spinning (`_mm_pause`) without end: when the thread of the block before was descheduled, the other threads spun and kept it off a CPU, so some runs took 2–8× as long (epyc chess_10m: user time 5 s instead of 2 s, the same system time; 11 of 80 runs > 2× the median). It now spins 2,048 pauses, then sleeps (`atomic::wait`, woken by the block before). On epyc2, 40 runs each of chess_10m (48 threads): CART 0.155 ± 0.131 s → 0.087 ± 0.005 s, C4.5 0.186 ± 0.163 s → 0.095 ± 0.005 s, no run above 2× the median. Only the parallel backend has those blocks (CUDA hybrid gives the CPU nodes under 4,096 rows; --serial and --simple have no pool), so only the parallel epyc runs (both sockets, one socket) start over, with 10 runs a case (full ECBDL 5), into `cluster-<dataset>-fix/`, replacing the old folders when done. The dell runs of ./tree_cpu run again too (below); until the reruns are in, the bars of old builds are marked (hatched, ⚠, from `problems.json`), as are runs more than twice the mean of the other runs.
- **One build for every epyc ./tree_cpu run (2026-10-28, `bench/cluster/submit-final-2026-10-28.sh`):** after the spin fix, C4.5's list of distinct values (its thresholds are training values) was made allocation-free: counted, then written in place on the pool, and in one pass with --serial (SOREL's C4.5 presort on epyc varied 11–34 s; see cluster-sorel). Same trees as before in every check (SOREL, HIGGS, airline, ECBDL 15M, serial and parallel). So that every epyc run of ./tree_cpu is of that one build, the parallel runs started before it (jobs 550–556) run again, and --serial runs again for every dataset (10 runs a case of about 2 minutes or less, 5 under about 10 minutes, longer ones once), into `cluster-<dataset>-fix/`, replacing the old folders. The long runs (full ECBDL and SOREL parallel, every --serial, then --simple) run one after another on epyc2, the short parallel ones on epyc1. CUDA hybrid, GPU only and --simple do not use that code and are not rerun. Every dell run of ./tree_cpu (parallel on both sockets, on one socket, dell2's 512 GB, --serial) runs again on the same build too (`bench/cluster/submit-dell-chess-2026-10-28.sh`, jobs 592–618: parallel on dell1, then --serial there; dell2 only its 512 GB runs), the same runs as before, replacing the old folders; --simple on dell stays.
- **Every run again on one build each, in batches (2026-10-28 evening, `bench/cluster/submit-batches-2026-10-28.sh`, jobs 636–645):** the pending jobs above were cancelled and replaced by a few batch jobs (`bench/cluster/batch.sh`: one SLURM job per node and group, its runs one after another). CUDA hybrid and GPU only on hopper and ampere run again on the new ./tree (67495f9, the CUDA rework: the smaller side's goes-left bitmap, the split column by position, upload overlapped with the presort, root counts on the GPU), installed by job 636 only after its trees matched the old build's (HIGGS, airline, ECBDL 15M; CART and C4.5, both GPU backends; 10–25 % faster there). On the CPU, dell and epyc for every backend (--simple, --serial, parallel on one socket, on both), on the final ./tree_cpu; the runs already done on it are kept, as are --simple's (that code did not change). Runs per case: about 2 minutes or less 10, under about 10 minutes 3, longer once; long runs on epyc2 (epyc1 stays free for the short ones), every dell run on dell1, short batch first. Test accuracy is now predicted once per distinct tree (`bench/run.py`), not once per run. The chart groups the bars by backend, the nodes under each; it reads `cluster-mnist8m-fix/` too: a rerun done replaces the old bar, one pending marks it ("old build; rerun queued") or shows as "queued".

## Results

Run on 2026-10-27/28 at 8187eb9 (`-dirty`: the uncommitted bench/ changes adding the dataset; ./tree's code is 8187eb9). The tables give means of ./tree's training time. Accuracies, nodes and depth are the same in every run.

![chart](chart.png)

- **Same tree everywhere:** every backend grew the same tree in every case.
- **Speed:** the H100 takes 4.1–4.3 s, epyc's parallel run 5.6–6.0 s (1.3–1.5× slower), the A100 5.8–6.2 s, dell 9.0 s. 784 features keep the CPU close to the GPUs.
- **One socket:** 1.9× epyc's two-socket time (10.4–11.3 s), 1.7× on dell (15–16 s).
- **Serial and simple:** --serial takes 218–240 s on epyc and 287–308 s on dell, about 40× (epyc) and 33× (dell) the parallel run. --simple takes 1,875–2,159 s, 7–9× --serial, with a third of the RAM (24 GiB).
- **Accuracy:** CART 84.5% on the test set, C4.5 74.7%. Test accuracy is above training accuracy: the training digits are distorted copies, harder than the original test digits.
- **Memory:** VRAM 49 GiB. GPU only needs 24 GiB of RAM, hybrid 72 GiB, the parallel CPU runs 73–74 GiB, --serial 71 GiB, --simple 24 GiB.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 84.51% | 80.77% | 2,047 | 10 |
| C4.5, no pruning | 74.65% | 70.96% | 1,453 | 10 |

| Run | CART | C4.5 | RAM GiB (peak) | VRAM GiB (peak) |
|---|---|---|---|---|
| hopper, CUDA hybrid | 4.08 s | 4.24 s | 71.7 | 48.9–49.2 |
| hopper, GPU only | 4.1 s | 4.28 s | 24.4 | 49.1–49.2 |
| ampere, CUDA hybrid | 5.79 s | 5.99 s | 71.7 | 48.8–49.2 |
| ampere, GPU only | 5.91 s | 6.2 s | 24.4 | 49.2 |
| epyc, parallel, both sockets | 5.97 s | 5.56 s | 74.0 | – |
| dell, parallel, both sockets | 9.03 s | 9 s | 74.0 | – |
| epyc, parallel, one socket | 11.3 s | 10.4 s | 72.6 | – |
| dell, parallel, one socket | 15.8 s | 14.9 s | 72.6 | – |
| epyc, --serial | 240 s | 218 s | 71.2 | – |
| dell, --serial | 308 s | 287 s | 71.2 | – |
| epyc, --simple | 1,875 s | 1,966 s | 23.8–23.9 | – |
| dell, --simple | 2,117 s | 2,159 s | 23.8–23.9 | – |
