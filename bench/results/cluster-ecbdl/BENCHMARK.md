# ECBDL'14 (631 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** `ecbdl`: all 31,992,921 training rows, on the CPU nodes only (the GPUs would need about 160 GB); `ecbdl_15m`: the first 15,000,000, on every node.
- **Test rows:** the competition's labelled test set, 2,897,917 rows. Until 2026-10-23 100,000 random training rows were the test set: cluster-ecbdl-old (removed; last in f4668df).
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning. The pruned runs of 2026-10-23 (CART α = 1e-5, C4.5 CF 0.25) are in git at 96705a7.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB), on `ecbdl_15m` only (the full set needs about 160 GB of VRAM); on epyc (2× EPYC 9255) and dell1 (2× Xeon Gold 6542Y), on both sets: parallel on both sockets, parallel on one socket, --serial and --simple (one thread). Runs per case: 10 when a run takes about 2 minutes or less, 3 under about 10 minutes, longer ones once (the full set's parallel runs, about 4 minutes with reading the data: 3; `--serial` CART on epyc, 503 s of training: 3).
- **Builds:** every run on one build of main, ./tree at ec77727 (with the CPU presort of 4fb419c and `-falign-functions=64 -falign-loops=32`), 2026-10-29/30, except --simple where it was not rerun: those runs are from 2026-10-27/28 on 67495f9 or 8187eb9 (`machine.json` says which; --simple's code has not changed since). Why, and the batches that ran them: [bench/cluster/README.md](../../cluster/README.md#reruns-of-2026-102930). Older runs are in git.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart-ecbdl.png` (full) and `chart-ecbdl_15m.png`, `bench/testset_chart.py` (log scale).

## Results

Run on 2026-10-29/30 (--simple on the dates above). The tables give the mean ± standard deviation of ./tree's training time (`bench/testset_chart.py --table`). Accuracies, nodes and depth are the same in every run. `simple-epyc-ecbdl_15m` holds the --simple runs of cluster-cpu-vs-others (2026-10-27, the same command).

![ECBDL'14, all training rows](chart-ecbdl.png)

![ECBDL'14, first 15M training rows](chart-ecbdl_15m.png)

- **Same tree everywhere:** on each training set, every backend grew the same tree in every run.
- **Accuracy is not informative:** only about 1.7% of the test rows are contacts, so the majority class alone scores 98.32%. The trees match or barely miss that score. A class-weighted or AUC measure would be needed to compare them.
- **15M sample:** the H100 takes 2.97–2.99 s (hybrid and GPU only alike), 2.1–2.6× faster than epyc (6.32 s and 7.81 s); the A100 5.0–5.1 s, 1.2–1.6× faster than epyc; dell 10.1–12.4 s, 1.6× epyc. GPU only needs 36 GiB of RAM, against 107 GiB for hybrid and 109–111 GiB for the parallel CPU runs.
- **One socket:** 1.8–1.9× the two-socket time on epyc, 1.7–1.8× on dell, on both sets: ECBDL's 631 features keep every core busy.
- **Full set (CPU only):** epyc takes 13.5 s and 17.6 s, dell 21.1 s and 27.9 s (1.6× epyc), with 238 GiB of RAM: 2.1–2.3× the 15M sample's time for 2.1× the rows.
- **Serial and simple:** --serial takes 34–38× the parallel run on both nodes (full set: 503 s and 623 s on epyc, 805 s and 980 s on dell). --simple takes 5,857–7,089 s on the full set, 11–12× --serial on epyc and 7–8× on dell, with a third of the RAM (76 GiB).

### ecbdl

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 98.30% | 97.86% | 1,917 | 10 |
| C4.5, no pruning | 98.32% | 97.86% | 785 | 10 |

| Run | runs | CART | C4.5 | RAM GiB (peak) |
|---|---|---|---|---|
| epyc, --simple, one thread | 1 | 5,857 s | 6,890 s | 75.8 |
| dell, --simple, one thread | 1 | 6,207 s | 7,089 s | 75.8 |
| epyc, --serial, one thread | 3 / 1 | 503 s ± 3.07 s | 623 s | 226.4 |
| dell, --serial, one thread | 1 | 805 s | 980 s | 226.4 |
| epyc, parallel, one socket | 3 | 25.9 s ± 321 ms | 32.4 s ± 140 ms | 231.9 |
| dell, parallel, one socket | 3 | 37.9 s ± 54.4 ms | 47.5 s ± 385 ms | 231.9 |
| epyc, parallel, both sockets | 3 | 13.5 s ± 33 ms | 17.6 s ± 64.4 ms | 237.6 |
| dell, parallel, both sockets | 3 | 21.1 s ± 13.2 ms | 27.9 s ± 71.5 ms | 237.6 |

### ecbdl_15m

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 98.31% | 97.92% | 1,863 | 10 |
| C4.5, no pruning | 98.32% | 97.92% | 725 | 10 |

| Run | runs | CART | C4.5 | RAM GiB (peak) | VRAM GiB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 1 | 2,273 s | 2,587 s | 35.6 | – |
| dell, --simple, one thread | 1 | 2,527 s | 2,815 s | 35.6 | – |
| epyc, --serial, one thread | 3 | 225 s ± 117 ms | 277 s ± 230 ms | 106.2 | – |
| dell, --serial, one thread | 3 | 351 s ± 176 ms | 425 s ± 169 ms | 106.2 | – |
| epyc, parallel, one socket | 10 | 11.8 s ± 27.3 ms | 14.4 s ± 24.1 ms | 108.8 | – |
| dell, parallel, one socket | 10 | 17.8 s ± 72.3 ms | 21.6 s ± 57.6 ms | 108.8 | – |
| epyc, parallel, both sockets | 10 | 6.32 s ± 13.1 ms | 7.81 s ± 32 ms | 111.4 | – |
| dell, parallel, both sockets | 10 | 10.1 s ± 119 ms | 12.4 s ± 43.4 ms | 111.4 | – |
| hopper, CUDA hybrid | 10 | 2.97 s ± 4.84 ms | 2.99 s ± 4.32 ms | 106.5 | 73.3 |
| ampere, CUDA hybrid | 10 | 5.07 s ± 34.7 ms | 5.1 s ± 28.7 ms | 106.5 | 73.3 |
| hopper, CUDA, GPU only | 10 | 2.97 s ± 4.3 ms | 2.99 s ± 5.07 ms | 36.0 | 73.3 |
| ampere, CUDA, GPU only | 10 | 5.02 s ± 28.7 ms | 5.05 s ± 30.1 ms | 36.0 | 73.3 |
