# SUSY (18 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** the first 4,500,000 rows.
- **Test rows:** the last 500,000, UCI's test set.
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning.
- **Runs:** CUDA `--hybrid` and `--gpu` (GPU only) on hopper (H100) and ampere (A100 80 GB), with every CPU of the node; on epyc (2× EPYC 9255) and dell (2× Xeon Gold 6542Y): `--multicore` on both sockets (48 threads), `--serial` and `--simple` (one thread). Runs per case: 5, `--simple` 10.
- **Builds:** ./tree at f53347a9 (`src/` = d2957ff2: presort of big columns in cache, GPU tiles without a cut skipped, CART bound per node), one build for every node, run on 2026-11-15/16 by `bench/cluster/final.sh -m 5` ([bench/cluster/README.md](../../cluster/README.md#the-final-rerun)): `--multicore` and CUDA on epyc1, dell1, hopper and ampere, `--serial` spread over epyc1/epyc2 and dell1/dell2/dell3 (the same hardware). `--simple` was not rerun, its code has not changed: its runs are from 2026-11-10/12, f1fe270e on epyc and ce907ddf on dell (`machine.json` says which). The runs before (f1fe270e, 10 runs per case, 2026-11-09/12) are in git at 1e655067.
- **Measured:** ./tree's training time (presort, build; not reading the CSV), peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes, depth and SHA-256.
- **Chart:** `chart.png`, `bench/testset_chart.py` (log scale).

## Results

Every run of a case grew the same tree (the same `tree_sha256`). The tables give the mean ± standard deviation of ./tree's training time (`bench/testset_chart.py --table`); "runs" is the number of runs of the case, also on the chart.

![chart](chart.png)

- **Speed:** both GPUs take 62–85 ms, a near-fixed cost at 4.5M rows (the A100 within 6–18% of the H100). That is 2.0–2.6× faster than epyc's `--multicore` (162 and 172 ms); dell takes 275 and 293 ms, 1.7× epyc.
- **Serial and simple:** `--serial` takes 2.60 s (CART) and 2.48 s (C4.5) on epyc, 14–16× `--multicore`, and 3.91 s and 3.95 s on dell. `--simple`, the reference, takes 70–80 s: 27–31× `--serial` on epyc, 19–20× on dell.
- **Against the runs before (f1fe270e):** `--multicore` 31–33% faster on epyc and 22% on dell, the GPUs 17–29%, `--serial` 8–22% on epyc and 3–12% on dell. The 100–300 ms runs spread up to ±15% (`--multicore` on epyc ±25 ms).
- **Accuracy:** CART reaches 79.0% on the test set, C4.5 78.7%.
- **Memory:** VRAM 1.6–1.7 GB; RAM at most 1.6 GB everywhere (GPU only and `--serial` 1.0 GB, `--simple` 0.5 GB).

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 79.01% | 79.16% | 1,959 | 10 |
| C4.5, no pruning | 78.65% | 78.74% | 743 | 10 |

| Run | runs | CART | C4.5 | RAM GB (peak) | VRAM GB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 10 | 70.3 s ± 224 ms | 76.7 s ± 386 ms | 0.5 | – |
| dell, --simple, one thread | 10 | 75.1 s ± 86.9 ms | 79.6 s ± 15 ms | 0.5 | – |
| epyc, --serial, one thread | 5 | 2.6 s ± 12.5 ms | 2.48 s ± 12.1 ms | 1.0 | – |
| dell, --serial, one thread | 5 | 3.91 s ± 1.13 ms | 3.95 s ± 2.92 ms | 1.0 | – |
| epyc, parallel, both sockets | 5 | 162 ms ± 24.9 ms | 172 ms ± 26.8 ms | 1.6 | – |
| dell, parallel, both sockets | 5 | 275 ms ± 21.3 ms | 293 ms ± 12.2 ms | 1.6 | – |
| hopper, CUDA hybrid | 5 | 62.4 ms ± 6.49 ms | 75.2 ms ± 5.91 ms | 1.6 | 1.7 |
| ampere, CUDA hybrid | 5 | 73.4 ms ± 1.98 ms | 84.8 ms ± 2.4 ms | 1.6 | 1.6 |
| hopper, CUDA, GPU only | 5 | 64 ms ± 6.33 ms | 78.1 ms ± 5.32 ms | 1.0 | 1.7 |
| ampere, CUDA, GPU only | 5 | 66.9 ms ± 1.2 ms | 80.7 ms ± 0.471 ms | 1.0 | 1.6 |
