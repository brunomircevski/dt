# airline (US flights, 13 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** the flights of 1987–2007, 109,258,555 rows.
- **Test rows:** the flights of 2008 (January to October), 5,810,462 rows: every row from the first with Year ≥ 2008 (airline has no published test set; the rows are in date order, so the tree predicts flights later than those it learned from). Until 2026-10-23 a random 500,000 rows were the test set: [cluster-airline-old](../cluster-airline-old/BENCHMARK.md).
- **Protocols:** depth ≤ 10; CART without pruning and pruned at α = 1e-5, C4.5 without pruning and pruned at CF 0.25.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB), parallel on every CPU of both sockets on epyc1 and dell1; no serial runs. Three runs per case, averaged.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart.png`, `bench/testset_chart.py`.

## Results

Run on 2026-10-23 at 74a011f. Each case ran 3 times; the tables give means of ./tree's training time. Accuracies, nodes and depth are the same in every run.

![chart](chart.png)

- **Same tree everywhere:** every backend grew the same tree in every case.
- **Speed:** the H100 is 4.8–5.3× faster than epyc on CART and 6.1–6.7× on unpruned C4.5. C4.5's pruning runs on the host CPU and doubles a GPU run (0.67 → 1.41 s on hopper), so the pruned C4.5 gap shrinks to 3.1–3.2×.
- **Accuracy:** 66.2–66.6% on 2008 flights, about half a point below training accuracy. Pruning changes it by at most 0.03 points.
- **Memory:** GPU only needs 13.0 GiB of RAM at most, against 18.0 GiB for hybrid. VRAM peaks at 12.7–18.4 GiB. The CPU nodes need 26–28 GiB.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 66.55% | 67.10% | 2,047 | 10 |
| CART, α 1e-5 | 66.52% | 67.04% | 509 | 10 |
| C4.5, no pruning | 66.21% | 66.62% | 929 | 10 |
| C4.5, CF 0.25 | 66.22% | 66.62% | 815 | 10 |

| Run | CART, no pruning | CART, α 1e-5 | C4.5, no pruning | C4.5, CF 0.25 | RAM GiB (peak, range over protocols) | VRAM GiB (peak, range over protocols) |
|---|---|---|---|---|---|---|
| hopper, CUDA hybrid | 0.647 s | 0.593 s | 0.666 s | 1.41 s | 6.1–18.0 | 12.7–18.4 |
| hopper, GPU only | 0.59 s | 0.588 s | 0.61 s | 1.46 s | 6.1–13.0 | 12.7–18.4 |
| ampere, CUDA hybrid | 0.992 s | 0.899 s | 0.995 s | 1.85 s | 6.1–18.0 | 12.7–18.4 |
| ampere, GPU only | 0.904 s | 0.901 s | 0.922 s | 1.79 s | 6.1–13.0 | 12.7–18.4 |
| epyc, parallel | 3.13 s | 3.04 s | 4.09 s | 4.55 s | 26.5–28.4 | – |
| dell, parallel | 4.47 s | 4.42 s | 6.09 s | 6.91 s | 26.0–27.8 | – |
