# SUSY (18 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** the first 4,500,000 rows.
- **Test rows:** the last 500,000, UCI's test set.
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning. The pruned runs of 2026-10-23 (CART α = 1e-5, C4.5 CF 0.25) are in git at 96705a7.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB); on epyc (2× EPYC 9255) and dell1 (2× Xeon Gold 6542Y): parallel on both sockets, parallel on one socket, --serial and --simple (one thread). Runs per case: 10 when a run takes about 2 minutes or less, 3 under about 10 minutes, longer ones once (the chart and the table give the count).
- **Builds:** every run on one build of main, ./tree at ec77727 (with the CPU presort of 4fb419c and `-falign-functions=64 -falign-loops=32`), 2026-10-29/30, except --simple where it was not rerun: those runs are from 2026-10-27/28 on 67495f9 or 8187eb9 (`machine.json` says which; --simple's code has not changed since). Why, and the batches that ran them: [bench/cluster/README.md](../../cluster/README.md#reruns-of-2026-102930). Older runs are in git.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart.png`, `bench/testset_chart.py` (log scale).

## Results

Run on 2026-10-29/30 (--simple on the dates above). The tables give the mean ± standard deviation of ./tree's training time (`bench/testset_chart.py --table`). Accuracies, nodes and depth are the same in every run.

![chart](chart.png)

- **Same tree everywhere:** every backend grew the same tree in every run.
- **Speed:** both GPUs take 82–119 ms, a near-fixed cost at 4.5M rows (the A100 matches the H100). That is 2.2–2.9× faster than epyc (0.24–0.25 s); dell takes 0.34–0.37 s.
- **One socket:** 1.11–1.18× the two-socket time on epyc, 1.19–1.25× on dell: 4.5M rows hardly fill two sockets. dell's one-socket time moves between batches on the same build: a rerun the next day took 0.36 s and 0.39 s against this table's 0.43 and 0.44 s ([cluster-cpu-scaling](../cluster-cpu-scaling/BENCHMARK.md#dell1-two-oddities-checked-on-2026-10-30)).
- **Serial and simple:** --serial takes 2.8 s and 3.1 s on epyc, 12× the parallel run, and 4.0 s and 4.4 s on dell. --simple takes 71–81 s on both: 25× --serial on epyc, 18–19× on dell.
- **Accuracy:** CART reaches 79.0% on the test set, C4.5 78.6%.
- **Memory:** VRAM 1.6–1.7 GiB; RAM at most 1.6 GiB everywhere (GPU only 1.0 GiB, --serial 1.1 GiB, --simple 0.5 GiB).

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 79.01% | 79.16% | 1,959 | 10 |
| C4.5, no pruning | 78.65% | 78.74% | 743 | 10 |

| Run | runs | CART | C4.5 | RAM GiB (peak) | VRAM GiB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 3 | 70.8 s ± 23.6 ms | 77.2 s ± 58.3 ms | 0.5 | – |
| dell, --simple, one thread | 10 | 76.3 s ± 118 ms | 80.7 s ± 27.9 ms | 0.5 | – |
| epyc, --serial, one thread | 10 | 2.84 s ± 18.6 ms | 3.1 s ± 21 ms | 1.1 | – |
| dell, --serial, one thread | 10 | 4.03 s ± 21.2 ms | 4.42 s ± 26.9 ms | 1.1 | – |
| epyc, parallel, one socket | 10 | 281 ms ± 3.09 ms | 277 ms ± 5.07 ms | 1.6 | – |
| dell, parallel, one socket | 10 | 425 ms ± 11.5 ms | 437 ms ± 7.5 ms | 1.6 | – |
| epyc, parallel, both sockets | 10 | 239 ms ± 14 ms | 249 ms ± 24.4 ms | 1.6 | – |
| dell, parallel, both sockets | 10 | 339 ms ± 10.1 ms | 368 ms ± 8.23 ms | 1.6 | – |
| hopper, CUDA hybrid | 10 | 83.7 ms ± 4.25 ms | 112 ms ± 2.61 ms | 1.6 | 1.7 |
| ampere, CUDA hybrid | 10 | 90.8 ms ± 3.35 ms | 119 ms ± 2.97 ms | 1.6 | 1.6 |
| hopper, CUDA, GPU only | 10 | 81.6 ms ± 3.82 ms | 109 ms ± 5.64 ms | 1.0 | 1.7 |
| ampere, CUDA, GPU only | 10 | 89.1 ms ± 1.64 ms | 107 ms ± 1.74 ms | 1.0 | 1.6 |
