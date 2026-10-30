# HIGGS (28 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** the first 10,500,000 rows.
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
- **Speed:** the H100 takes 0.15–0.17 s, 3.5–4.0× faster than epyc's parallel run (0.59–0.61 s); the A100 0.21–0.25 s. dell is 1.4–1.7× slower than epyc.
- **One socket:** 1.4–1.6× the two-socket time on epyc, 1.5–1.7× on dell. dell's one-socket time moves between batches on the same build: a rerun the next day took 1.06–1.07 s (CART) and 1.45–1.47 s (C4.5) against this table's 1.21 and 1.68 s, and the previous build was no faster when alternated with this one ([cluster-cpu-scaling](../cluster-cpu-scaling/BENCHMARK.md#dell1-two-oddities-checked-on-2026-10-30)).
- **Serial and simple:** --serial takes 10.4 s (CART) and 11.9 s (C4.5) on epyc, 17–20× the parallel run, and 15.6 s and 20.3 s on dell. --simple, the reference, takes 236–258 s on both: 22–23× --serial on epyc, 13–15× on dell.
- **Accuracy:** CART reaches 70.5% on the test set, C4.5 63.7%.
- **Memory:** VRAM 3.3 GiB. GPU only needs 1.8 GiB of RAM, hybrid 4.0 GiB, the parallel CPU runs 5.4–5.6 GiB, --serial 3.6 GiB, --simple 1.4 GiB.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 70.52% | 70.63% | 2,029 | 10 |
| C4.5, no pruning | 63.73% | 63.76% | 935 | 10 |

| Run | runs | CART | C4.5 | RAM GiB (peak) | VRAM GiB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 3 | 236 s ± 196 ms | 257 s ± 103 ms | 1.4 | – |
| dell, --simple, one thread | 3 | 241 s ± 84.6 ms | 258 s ± 108 ms | 1.4 | – |
| epyc, --serial, one thread | 10 | 10.4 s ± 23 ms | 11.9 s ± 23.4 ms | 3.6 | – |
| dell, --serial, one thread | 10 | 15.6 s ± 104 ms | 20.3 s ± 79.8 ms | 3.6 | – |
| epyc, parallel, one socket | 10 | 829 ms ± 11.3 ms | 925 ms ± 6.1 ms | 5.4 | – |
| dell, parallel, one socket | 10 | 1.21 s ± 22.2 ms | 1.68 s ± 46.3 ms | 5.4 | – |
| epyc, parallel, both sockets | 10 | 607 ms ± 19.2 ms | 590 ms ± 18.1 ms | 5.6 | – |
| dell, parallel, both sockets | 10 | 823 ms ± 34.8 ms | 993 ms ± 36.7 ms | 5.6 | – |
| hopper, CUDA hybrid | 10 | 151 ms ± 2.89 ms | 170 ms ± 1.7 ms | 4.0 | 3.3 |
| ampere, CUDA hybrid | 10 | 210 ms ± 1.58 ms | 250 ms ± 3.44 ms | 4.0 | 3.3 |
| hopper, CUDA, GPU only | 10 | 150 ms ± 2.14 ms | 164 ms ± 3.74 ms | 1.8 | 3.3 |
| ampere, CUDA, GPU only | 10 | 205 ms ± 2 ms | 218 ms ± 1.76 ms | 1.8 | 3.3 |
