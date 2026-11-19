# HIGGS (28 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/tools/cluster/README.md](../tools/cluster/README.md#test-set-benchmarks)).

- **Training rows:** the first 10,500,000 rows.
- **Test rows:** the last 500,000, UCI's test set.
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning.
- **Runs:** CUDA `--hybrid` and `--gpu` (GPU only) on hopper (H100) and ampere (A100 80 GB), with every CPU of the node; on epyc (2× EPYC 9255) and dell (2× Xeon Gold 6542Y): `--multicore` on both sockets (48 threads), `--serial` and `--simple` (one thread). 10 runs a case, each a fresh process with no warm-up, in a shuffled, interleaved order.
- **Builds:** ./tree at 8fe0c73d (CPU runs) and 713914b8 (GPU runs: the same tree code plus `--gpu-startup`, the page-locked hand-off buffers made only for `--hybrid`, and always an 80 GB GPU), one build for every node, run on 2026-11-17 by `bench/tools/cluster/final.sh -D higgs,susy` (GPU runs: `-D higgs,susy,mnist8m -k gpu,gpulibs,e2e`) ([bench/tools/cluster/README.md](../tools/cluster/README.md#the-final-rerun)): `--multicore` on epyc1 and dell1, `--serial` on epyc2 and dell3, `--simple` on epyc1 and dell3, CUDA on hopper and ampere. The GPU runs time ./tree's GPU setup (kernel loading, memory pool, page-locked staging buffers, copy threads) by the rule every GPU tool follows: only the bare CUDA context is untimed (`--gpu-startup context`, [bench/tools/README.md](../tools/README.md#how-time-is-measured)). The 2026-11-15 runs (f53347a9, 5 runs a case) are in git at 7d906f7d. **Out of date:** `--multicore` on dell1 (⚠, faded on the chart): from 2026-11-17 the dell runs are on a 512 GB node (dell2–4, half dell1's memory bandwidth; [bench/tools/cluster/README.md](../tools/cluster/README.md)), so dell's `--multicore` waits for the rerun on dell4.
- **Measured:** ./tree's training time (presort, build; not reading the CSV), peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes, depth and SHA-256.
- **Chart:** `chart.png`, `bench/tools/testset_chart.py` (log scale).

## Results

Every run of a case grew the same tree (the same `tree_sha256`). The tables give the mean of ./tree's training time (`bench/tools/testset_chart.py --table`); "runs" is the number of runs of the case, also on the chart.

![chart](chart.png)

- **Speed:** GPU only on the H100 takes 221 ms (CART) and 241 ms (C4.5): 155–157 ms of GPU setup (kernel loading, memory pool, staging buffers; a fixed cost) and 66–83 ms of build, 1.5–1.6× faster than epyc's `--multicore` (348 and 366 ms). The A100 takes 247 and 274 ms. Hybrid's setup also makes its 256 MB of page-locked hand-off buffers (259–300 ms of setup), so at depth 10 hybrid (326–348 ms on the H100, 392–417 ms on the A100) is slower than GPU only and no faster than epyc. dell takes 568 and 677 ms, 1.6–1.9× epyc.
- **Serial and simple:** `--serial` takes 7.89 s (CART) and 7.42 s (C4.5) on epyc, 20–23× `--multicore`, and 12.9 s and 12.6 s on dell. `--simple` takes 237–259 s on both: 30–35× `--serial` on epyc, 19–20× on dell.
- **Against the 2026-11-15 runs (7d906f7d):** the CPU runs are within ±7% (the same tree code). The GPU runs take more by the setup that was untimed before (see Builds): GPU only 81–92 ms more, hybrid 194–223 ms.
- **Accuracy:** CART reaches 70.5% on the test set, C4.5 63.7%.
- **Memory:** VRAM 3.3 GB. GPU only needs 1.5 GB of RAM, hybrid 4.0 GB, `--multicore` 5.7 GB, `--serial` 3.5 GB, `--simple` 1.4 GB.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 70.52% | 70.63% | 2,029 | 10 |
| C4.5, no pruning | 63.73% | 63.76% | 935 | 10 |

| Run | runs | CART | C4.5 | RAM GB (peak) | VRAM GB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 10 | 237 s | 259 s | 1.4 | – |
| dell, --simple, one thread | 10 | 241 s | 258 s | 1.4 | – |
| epyc, --serial, one thread | 10 | 7.89 s | 7.42 s | 3.5 | – |
| dell, --serial, one thread | 10 | 12.9 s | 12.6 s | 3.5 | – |
| epyc, parallel, both sockets | 10 | 348 ms | 366 ms | 5.7 | – |
| dell, parallel, both sockets | 10 | 568 ms | 677 ms | 5.7 | – |
| hopper, CUDA hybrid | 10 | 326 ms | 348 ms | 4.0 | 3.3 |
| ampere, CUDA hybrid | 10 | 392 ms | 417 ms | 4.0 | 3.3 |
| hopper, CUDA, GPU only | 10 | 221 ms | 241 ms | 1.5 | 3.3 |
| ampere, CUDA, GPU only | 10 | 247 ms | 274 ms | 1.5 | 3.3 |
