# HIGGS (28 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** the first 10,500,000 rows.
- **Test rows:** the last 500,000, UCI's test set.
- **Protocols:** depth ≤ 10; CART without pruning and pruned at α = 1e-5, C4.5 without pruning and pruned at CF 0.25.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB), parallel on every CPU of both sockets on epyc1 and dell1; no serial runs. Three runs per case, averaged.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart.png`, `bench/testset_chart.py`.

## Results

Run on 2026-10-23 at 74a011f. Each case ran 3 times; the tables give means of ./tree's training time. Accuracies, nodes and depth are the same in every run.

![chart](chart.png)

- **Same tree everywhere:** every backend grew the same tree in every case.
- **Speed:** the H100 takes 0.17–0.34 s, 2.0–3.6× faster than epyc (0.61–0.67 s). dell is 1.3–1.7× slower than epyc.
- **Accuracy:** CART reaches 70.5% on the test set, C4.5 63.7%. Pruning costs at most 0.05 points and cuts CART from 2,029 to 585 nodes.
- **Memory:** VRAM 3.3–4.5 GiB. GPU only needs 1.7–3.1 GiB of RAM, hybrid up to 4.1 GiB; the CPU nodes 5.6–5.9 GiB.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 70.52% | 70.63% | 2,029 | 10 |
| CART, α 1e-5 | 70.47% | 70.56% | 585 | 10 |
| C4.5, no pruning | 63.73% | 63.76% | 935 | 10 |
| C4.5, CF 0.25 | 63.73% | 63.76% | 779 | 10 |

| Run | CART, no pruning | CART, α 1e-5 | C4.5, no pruning | C4.5, CF 0.25 | RAM GiB (peak, range over protocols) | VRAM GiB (peak, range over protocols) |
|---|---|---|---|---|---|---|
| hopper, CUDA hybrid | 0.176 s | 0.17 s | 0.189 s | 0.313 s | 1.8–4.1 | 3.3–4.5 |
| hopper, GPU only | 0.175 s | 0.169 s | 0.192 s | 0.335 s | 1.8–3.1 | 3.3–4.5 |
| ampere, CUDA hybrid | 0.242 s | 0.23 s | 0.268 s | 0.385 s | 1.8–4.1 | 3.3–4.4 |
| ampere, GPU only | 0.23 s | 0.23 s | 0.246 s | 0.382 s | 1.7–3.1 | 3.3–4.4 |
| epyc, parallel | 0.641 s | 0.615 s | 0.61 s | 0.665 s | 5.6–5.9 | – |
| dell, parallel | 0.832 s | 0.829 s | 1.01 s | 1.1 s | 5.6–5.9 | – |
