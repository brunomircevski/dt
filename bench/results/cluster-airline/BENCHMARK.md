# airline (US flights, 13 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** the flights of 1987–2007, 109,258,555 rows.
- **Test rows:** the flights of 2008 (January to October), 5,810,462 rows: every row from the first with Year ≥ 2008 (airline has no published test set; the rows are in date order, so the tree predicts flights later than those it learned from). Until 2026-10-23 a random 500,000 rows were the test set: cluster-airline-old (removed; last in f4668df).
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning. The pruned runs of 2026-10-23 (CART α = 1e-5, C4.5 CF 0.25) are in git at 96705a7.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB); on epyc (2× EPYC 9255) and dell1 (2× Xeon Gold 6542Y): parallel on both sockets, parallel on one socket, --serial and --simple (one thread). Runs per case: 10 when a run takes about 2 minutes or less, 3 under about 10 minutes, longer ones once (the chart and the table give the count).
- **Builds:** every run on one build of main, ./tree at ec77727 (with the CPU presort of 4fb419c and `-falign-functions=64 -falign-loops=32`), 2026-10-29/30, except --simple where it was not rerun: those runs are from 2026-10-27/28 on 67495f9 or 8187eb9 (`machine.json` says which; --simple's code has not changed since). Why, and the batches that ran them: [bench/cluster/README.md](../../cluster/README.md#reruns-of-2026-102930). Older runs are in git.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart.png`, `bench/testset_chart.py` (log scale).

## Results

Run on 2026-10-29/30 (--simple on the dates above). The tables give the mean ± standard deviation of ./tree's training time (`bench/testset_chart.py --table`). Accuracies, nodes and depth are the same in every run.

![chart](chart.png)

- **Same tree everywhere:** every backend grew the same tree in every run.
- **Speed:** the H100 takes 0.49–0.51 s (hybrid and GPU only alike), 5.8–7.6× faster than epyc (2.87 s and 3.83 s). The A100 takes 0.80–0.82 s, 3.6–4.7× faster than epyc; dell 4.2–5.7 s, 1.5× epyc.
- **One socket:** 1.10–1.15× the two-socket time on epyc, 1.02–1.05× on dell.
- **Serial and simple:** --serial takes 35 s and 44 s on epyc, 11–12× the parallel run, and 59 s and 73 s on dell. --simple takes 770–849 s on both: 19–22× --serial on epyc, 12–14× on dell.
- **Accuracy:** 66.2–66.6% on 2008 flights, about half a point below training accuracy.
- **Memory:** GPU only needs 6.1 GiB of RAM, hybrid 16.8 GiB; VRAM 12.7 GiB. The parallel CPU runs need 26.0–26.3 GiB, --serial 18.6 GiB, --simple 7.4 GiB.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 66.55% | 67.10% | 2,047 | 10 |
| C4.5, no pruning | 66.21% | 66.62% | 929 | 10 |

| Run | runs | CART | C4.5 | RAM GiB (peak) | VRAM GiB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 3 | 770 s ± 787 ms | 813 s ± 332 ms | 7.4 | – |
| dell, --simple, one thread | 1 | 807 s | 849 s | 7.4 | – |
| epyc, --serial, one thread | 10 | 35 s ± 36.7 ms | 43.9 s ± 45.7 ms | 18.6 | – |
| dell, --serial, one thread | 10 | 58.9 s ± 199 ms | 72.7 s ± 174 ms | 18.6 | – |
| epyc, parallel, one socket | 10 | 3.31 s ± 22.9 ms | 4.22 s ± 60.1 ms | 26.0 | – |
| dell, parallel, one socket | 10 | 4.43 s ± 140 ms | 5.8 s ± 197 ms | 26.0 | – |
| epyc, parallel, both sockets | 10 | 2.87 s ± 37.4 ms | 3.83 s ± 68.4 ms | 26.0 | – |
| dell, parallel, both sockets | 10 | 4.21 s ± 90.3 ms | 5.68 s ± 143 ms | 26.0 | – |
| hopper, CUDA hybrid | 10 | 493 ms ± 2.52 ms | 507 ms ± 4.43 ms | 16.8 | 12.7 |
| ampere, CUDA hybrid | 10 | 806 ms ± 2.8 ms | 817 ms ± 4.58 ms | 16.8 | 12.7 |
| hopper, CUDA, GPU only | 10 | 495 ms ± 3.81 ms | 508 ms ± 5.91 ms | 6.1 | 12.7 |
| ampere, CUDA, GPU only | 10 | 796 ms ± 3.37 ms | 807 ms ± 3.82 ms | 6.1 | 12.7 |
