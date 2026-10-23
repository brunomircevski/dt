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

Queued on 2026-10-23; not yet run.
