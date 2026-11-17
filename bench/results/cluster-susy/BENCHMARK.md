# SUSY (18 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** the first 4,500,000 rows.
- **Test rows:** the last 500,000, UCI's test set.
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning.
- **Runs:** CUDA `--hybrid` and `--gpu` (GPU only) on hopper (H100) and ampere (A100 80 GB), with every CPU of the node; on epyc (2× EPYC 9255) and dell (2× Xeon Gold 6542Y): `--multicore` on both sockets (48 threads), `--serial` and `--simple` (one thread). 10 runs a case, each a fresh process with no warm-up, in a shuffled, interleaved order.
- **Builds:** ./tree at 8fe0c73d (CPU runs) and 713914b8 (GPU runs: the same tree code plus `--gpu-startup`, the page-locked hand-off buffers made only for `--hybrid`, and always an 80 GB GPU), one build for every node, run on 2026-11-17 by `bench/cluster/final.sh -D higgs,susy` (GPU runs: `-D higgs,susy,mnist8m -k gpu,gpulibs,e2e`) ([bench/cluster/README.md](../../cluster/README.md#the-final-rerun)): `--multicore` on epyc1 and dell1, `--serial` on epyc2 and dell3, `--simple` on epyc1 and dell3, CUDA on hopper and ampere. The GPU runs time ./tree's GPU setup (kernel loading, memory pool, page-locked staging buffers, copy threads) by the rule every GPU tool follows: only the bare CUDA context is untimed (`--gpu-startup context`, [bench/README.md](../../README.md#how-time-is-measured)). The 2026-11-15 runs (f53347a9, 5 runs a case) are in git at 7d906f7d. **Out of date:** `--multicore` on dell1 (⚠, faded on the chart): from 2026-11-17 the dell runs are on a 512 GB node (dell2–4, half dell1's memory bandwidth; [bench/cluster/README.md](../../cluster/README.md)), so dell's `--multicore` waits for the rerun on dell4.
- **Measured:** ./tree's training time (presort, build; not reading the CSV), peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes, depth and SHA-256.
- **Chart:** `chart.png`, `bench/testset_chart.py` (log scale).

## Results

Every run of a case grew the same tree (the same `tree_sha256`). The tables give the mean of ./tree's training time (`bench/testset_chart.py --table`); "runs" is the number of runs of the case, also on the chart.

![chart](chart.png)

- **Speed:** GPU only takes 150–167 ms on both GPUs, most of it the GPU setup (120–135 ms: kernel loading, memory pool, staging buffers, a fixed cost); the build takes 26–36 ms. That is 1.1–1.2× faster than epyc's `--multicore` (179 and 184 ms). Hybrid's setup also makes its 256 MB of page-locked hand-off buffers (238–262 ms of setup), so hybrid (267–302 ms) is 1.5–1.6× slower than epyc: at 4.5M rows the GPU's fixed setup outweighs its faster build. dell takes 206 and 238 ms, 1.15–1.3× epyc.
- **Serial and simple:** `--serial` takes 2.61 s (CART) and 2.49 s (C4.5) on epyc, 14–15× `--multicore`, and 3.92 s and 3.95 s on dell. `--simple`, the reference, takes 70–80 s: 27–31× `--serial` on epyc, 19–20× on dell.
- **Against the 2026-11-15 runs (7d906f7d):** `--serial` and `--simple` within 1%; `--multicore` 7–10% slower on epyc and 19–25% faster on dell (runs of about 0.2 s, 5 runs a case before, which spread up to ±15%). The GPU runs take more by the setup that was untimed before (see Builds): GPU only 83–89 ms more, hybrid 202–217 ms.
- **Accuracy:** CART reaches 79.0% on the test set, C4.5 78.7%.
- **Memory:** VRAM 1.6–1.7 GB; RAM at most 1.6 GB everywhere (GPU only 0.7 GB, `--serial` 1.0 GB, `--simple` 0.5 GB).

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 79.01% | 79.16% | 1,959 | 10 |
| C4.5, no pruning | 78.65% | 78.74% | 743 | 10 |

| Run | runs | CART | C4.5 | RAM GB (peak) | VRAM GB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 10 | 70.3 s | 76.9 s | 0.5 | – |
| dell, --simple, one thread | 10 | 75.1 s | 79.7 s | 0.5 | – |
| epyc, --serial, one thread | 10 | 2.61 s | 2.49 s | 1.0 | – |
| dell, --serial, one thread | 10 | 3.92 s | 3.95 s | 1.0 | – |
| epyc, parallel, both sockets | 10 | 179 ms | 184 ms | 1.6 | – |
| dell, parallel, both sockets | 10 | 206 ms | 238 ms | 1.6 | – |
| hopper, CUDA hybrid | 10 | 267 ms | 277 ms | 1.6 | 1.7 |
| ampere, CUDA hybrid | 10 | 287 ms | 302 ms | 1.6 | 1.6 |
| hopper, CUDA, GPU only | 10 | 150 ms | 167 ms | 0.7 | 1.7 |
| ampere, CUDA, GPU only | 10 | 150 ms | 164 ms | 0.7 | 1.6 |
