# ECBDL'14 (631 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** `ecbdl`: all 31,992,921 training rows, on the CPU nodes only (the GPUs would need about 160 GB); `ecbdl_15m`: the first 15,000,000, on every node.
- **Test rows:** the competition's labelled test set, 2,897,917 rows. Until 2026-10-23 100,000 random training rows were the test set: [cluster-ecbdl-old](../cluster-ecbdl-old/BENCHMARK.md).
- **Protocols:** depth ≤ 10; CART without pruning and pruned at α = 1e-5, C4.5 without pruning and pruned at CF 0.25.
- **Runs:** CUDA hybrid and GPU only on hopper (H100) and ampere (A100 80 GB), parallel on every CPU of both sockets on epyc1 and dell1; no serial runs. Three runs per case, averaged.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart-ecbdl.png` (full) and `chart-ecbdl_15m.png`, `bench/testset_chart.py`.

## Results

Queued on 2026-10-23; not yet run.
