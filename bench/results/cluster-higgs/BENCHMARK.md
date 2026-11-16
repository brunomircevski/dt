# HIGGS (28 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** the first 10,500,000 rows.
- **Test rows:** the last 500,000, UCI's test set.
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning.
- **Runs:** CUDA `--hybrid` and `--gpu` (GPU only) on hopper (H100) and ampere (A100 80 GB), with every CPU of the node; on epyc (2× EPYC 9255) and dell (2× Xeon Gold 6542Y): `--multicore` on both sockets (48 threads), `--serial` and `--simple` (one thread). Runs per case: 5, `--simple` 10.
- **Builds:** ./tree at f53347a9 (`src/` = d2957ff2: presort of big columns in cache, GPU tiles without a cut skipped, CART bound per node), one build for every node, run on 2026-11-15/16 by `bench/cluster/final.sh -m 5` ([bench/cluster/README.md](../../cluster/README.md#the-final-rerun)): `--multicore` and CUDA on epyc1, dell1, hopper and ampere, `--serial` spread over epyc1/epyc2 and dell1/dell2/dell3 (the same hardware). `--simple` was not rerun, its code has not changed: its runs are from 2026-11-10/12, f1fe270e on epyc and ce907ddf on dell (`machine.json` says which). The runs before (f1fe270e, 10 runs per case, 2026-11-09/12) are in git at 1e655067.
- **Measured:** ./tree's training time (presort, build; not reading the CSV), peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes, depth and SHA-256.
- **Chart:** `chart.png`, `bench/testset_chart.py` (log scale).

## Results

Every run of a case grew the same tree (the same `tree_sha256`). The tables give the mean ± standard deviation of ./tree's training time (`bench/testset_chart.py --table`); "runs" is the number of runs of the case, also on the chart.

![chart](chart.png)

- **Speed:** the H100 takes 129–154 ms, 2.2–2.5× faster than epyc's `--multicore` (327 and 342 ms); the A100 166–194 ms, 1.8–1.9× faster. dell takes 595 and 688 ms, 1.8–2.0× epyc. Hybrid and GPU only are within 2–5% of each other at depth 10.
- **Serial and simple:** `--serial` takes 7.90 s (CART) and 7.41 s (C4.5) on epyc, 22–24× `--multicore`, and 12.7 s and 12.6 s on dell. `--simple` takes 237–259 s on both: 30–35× `--serial` on epyc, 19–21× on dell.
- **Against the runs before (f1fe270e):** `--multicore` 44–46% faster on epyc and 28–45% on dell, `--serial` 24–39% on epyc and 18–45% on dell, the GPUs 8–20%.
- **Accuracy:** CART reaches 70.5% on the test set, C4.5 63.7%.
- **Memory:** VRAM 3.3 GB. GPU only needs 1.8 GB of RAM, hybrid 4.0 GB, `--multicore` 5.7 GB, `--serial` 3.5 GB, `--simple` 1.4 GB.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 70.52% | 70.63% | 2,029 | 10 |
| C4.5, no pruning | 63.73% | 63.76% | 935 | 10 |

| Run | runs | CART | C4.5 | RAM GB (peak) | VRAM GB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 10 | 237 s ± 966 ms | 259 s ± 1.43 s | 1.4 | – |
| dell, --simple, one thread | 10 | 241 s ± 13.7 ms | 258 s ± 25.7 ms | 1.4 | – |
| epyc, --serial, one thread | 5 | 7.9 s ± 6.13 ms | 7.41 s ± 17.9 ms | 3.5 | – |
| dell, --serial, one thread | 5 | 12.7 s ± 116 ms | 12.6 s ± 226 ms | 3.5 | – |
| epyc, parallel, both sockets | 5 | 327 ms ± 14 ms | 342 ms ± 4.96 ms | 5.7 | – |
| dell, parallel, both sockets | 5 | 595 ms ± 25.9 ms | 688 ms ± 15.7 ms | 5.7 | – |
| hopper, CUDA hybrid | 5 | 131 ms ± 5.3 ms | 154 ms ± 6.91 ms | 4.0 | 3.3 |
| ampere, CUDA hybrid | 5 | 175 ms ± 6.33 ms | 194 ms ± 2.63 ms | 4.0 | 3.3 |
| hopper, CUDA, GPU only | 5 | 129 ms ± 10.4 ms | 151 ms ± 8.03 ms | 1.8 | 3.3 |
| ampere, CUDA, GPU only | 5 | 166 ms ± 2.07 ms | 190 ms ± 2.23 ms | 1.8 | 3.3 |
