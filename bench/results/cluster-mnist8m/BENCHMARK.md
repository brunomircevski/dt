# MNIST8M (784 features, 10 classes): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** infinite MNIST's 8,100,000 distorted digits (LIBSVM `mnist8m`), 784 pixel features (0–255).
- **Test rows:** the original MNIST test set, 10,000 rows (LIBSVM `mnist.t`).
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB); on epyc (2× EPYC 9255) and dell1 (2× Xeon Gold 6542Y): parallel on both sockets, parallel on one socket, --serial and --simple (one thread). Runs per case: 10 when a run takes about 2 minutes or less, 3 under about 10 minutes, longer ones once (the chart and the table give the count).
- **Builds:** every run on one build of main, ./tree at ec77727 (with the CPU presort of 4fb419c and `-falign-functions=64 -falign-loops=32`), 2026-10-29/30, except --simple where it was not rerun: those runs are from 2026-10-27/28 on 67495f9 or 8187eb9 (`machine.json` says which; --simple's code has not changed since). Why, and the batches that ran them: [bench/cluster/README.md](../../cluster/README.md#reruns-of-2026-102930). Older runs are in git.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart.png`, `bench/testset_chart.py` (log scale).

## Results

Run on 2026-10-29/30 (--simple on the dates above). The tables give the mean ± standard deviation of ./tree's training time (`bench/testset_chart.py --table`). Accuracies, nodes and depth are the same in every run.

![chart](chart.png)

- **Same tree everywhere:** every backend grew the same tree in every run.
- **Speed:** the H100 takes 3.6–3.7 s, 1.5–1.7× faster than epyc's parallel run (5.6–5.9 s); the A100 5.2–5.5 s, 2–14% faster than epyc; dell 9.3–9.4 s. 784 features keep the CPU close to the GPUs.
- **One socket:** 1.8–1.9× epyc's two-socket time (10.3–11.1 s), 1.6–1.7× on dell (15.5–16.0 s).
- **Serial and simple:** --serial takes 209–234 s on epyc and 293–315 s on dell, 37–40× (epyc) and 31–34× (dell) the parallel run. --simple takes 1,875–2,159 s, 7–9× --serial, with a third of the RAM (24 GiB).
- **Accuracy:** CART 84.5% on the test set, C4.5 74.7%. Test accuracy is above training accuracy: the training digits are distorted copies, harder than the original test digits.
- **Memory:** VRAM 49 GiB. GPU only needs 24 GiB of RAM, hybrid 72 GiB, the parallel CPU runs 73–74 GiB, --serial 71 GiB, --simple 24 GiB.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 84.51% | 80.77% | 2,047 | 10 |
| C4.5, no pruning | 74.65% | 70.96% | 1,453 | 10 |

| Run | runs | CART | C4.5 | RAM GiB (peak) | VRAM GiB (peak) |
|---|---|---|---|---|---|
| epyc, --simple, one thread | 1 | 1,875 s | 1,966 s | 23.8 | – |
| dell, --simple, one thread | 1 | 2,117 s | 2,159 s | 23.8 | – |
| epyc, --serial, one thread | 3 | 234 s ± 896 ms | 209 s ± 253 ms | 71.2 | – |
| dell, --serial, one thread | 3 | 315 s ± 184 ms | 293 s ± 187 ms | 71.2 | – |
| epyc, parallel, one socket | 10 | 11.1 s ± 19.2 ms | 10.3 s ± 19 ms | 72.6 | – |
| dell, parallel, one socket | 10 | 16 s ± 238 ms | 15.5 s ± 193 ms | 72.6 | – |
| epyc, parallel, both sockets | 10 | 5.92 s ± 15.3 ms | 5.64 s ± 22.4 ms | 74.0 | – |
| dell, parallel, both sockets | 10 | 9.28 s ± 97.7 ms | 9.44 s ± 87.6 ms | 74.0 | – |
| hopper, CUDA hybrid | 10 | 3.56 s ± 8.39 ms | 3.68 s ± 13.7 ms | 71.7 | 49.0 |
| ampere, CUDA hybrid | 10 | 5.2 s ± 30.4 ms | 5.33 s ± 17.3 ms | 71.7 | 49.0 |
| hopper, CUDA, GPU only | 10 | 3.57 s ± 11.9 ms | 3.71 s ± 12.9 ms | 24.4 | 49.2 |
| ampere, CUDA, GPU only | 10 | 5.38 s ± 72.4 ms | 5.53 s ± 9.62 ms | 24.4 | 49.2 |
