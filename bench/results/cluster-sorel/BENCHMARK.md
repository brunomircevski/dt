# SOREL-20M (malware, 2381 features): training set and test set

./tree only, trained on the training rows and scored on the test rows
([bench/cluster/README.md](../../cluster/README.md#test-set-benchmarks)).

- **Training rows:** SOREL-20M's training set, 12,699,013 rows of EMBER v2 features (2381, float32), label malware/benign. The CSV is 136 GB; as float32 about 120 GB, so it does not fit an 80 GB GPU: CPU runs only.
- **Test rows:** SOREL-20M's test set, 4,195,042 rows.
- **Protocols:** depth ≤ 10, CART (gini) and C4.5 (gain ratio), both without pruning.
- **Runs:** on epyc (2× EPYC 9255, 512 GB) and dell1 (2× Xeon Gold 6542Y, 1 TB): parallel on both sockets, --serial and --simple (one thread); parallel on one socket on dell1 only, since one epyc socket has 256 GB and the runs need 338–354 GiB. No GPU runs: the data does not fit in 80 GB of VRAM. Runs per case: 10 when a run takes about 2 minutes or less, 3 under about 10 minutes, longer ones once, the parallel runs 3 (the chart and the table give the count).
- **Builds:** every run on one build of main, ./tree at ec77727 (with the CPU presort of 4fb419c and `-falign-functions=64 -falign-loops=32`), 2026-10-29/30, except --simple where it was not rerun: those runs are from 2026-10-27/28 on 67495f9 or 8187eb9 (`machine.json` says which; --simple's code has not changed since). Why, and the batches that ran them: [bench/cluster/README.md](../../cluster/README.md#reruns-of-2026-102930). Older runs are in git.
- **Measured:** ./tree's training time, peak RAM while training, peak VRAM; the tree's test and training accuracy, nodes and depth.
- **Chart:** `chart.png`, `bench/testset_chart.py` (log scale).

## Results

Run on 2026-10-29/30 (parallel and --serial) and 2026-10-27/28 (--simple). Parallel runs: 3 each, with about 30 minutes for the whole process (reading the 136 GB CSV); --serial and --simple once. The tables give ./tree's training time (`bench/testset_chart.py --table`), RAM the mean of CART and C4.5. Accuracies, nodes and depth are the same in every run.

![chart](chart.png)

- **Same tree everywhere:** every run grew the same tree.
- **Speed:** CART takes 21.8 s on epyc and 32.5 s on dell, C4.5 28.8 s and 44.3 s (dell 1.5× epyc). One socket on dell takes 1.7–1.8× the two-socket time (58.7 s and 77.2 s).
- **Serial:** 878 s (CART) and 1,130 s (C4.5) on epyc, 39–40× the parallel run; 1,295 s and 1,825 s on dell, 40–41×.
- **Simple:** 12,414 s (CART) and 12,712 s (C4.5) on dell, 14,124 s and 14,240 s on epyc (about 4 h): epyc is 1.1× slower here, unlike every other run. --simple is 13–16× --serial on epyc, 7–10× on dell. RAM: 113 GiB (CART) and 123 GiB (C4.5), a third of the parallel runs'.
- **Accuracy:** CART 92.7% on the test set, C4.5 78.1%.
- **Memory:** 338–343 GiB for CART, 350–354 GiB for C4.5, about 12 bytes for each of the 30 billion values. C4.5 peaked at 369–371 GiB in the runs before the 2026-10-28 build, whose C4.5 distinct values are counted, then written in place.
- **The harness once ran out of memory, not ./tree:** on 2026-10-28 C4.5 on epyc died at 361–362 GiB (two OOM kills in /proc/vmstat), because run.py's data check had read the 136 GB CSV with pyarrow in its own process and kept about 135 GiB. run.py now runs that check in a child process; free memory before each run equals that at the start of the job.

| Protocol | test acc | train acc | nodes | depth |
|---|---|---|---|---|
| CART, no pruning | 92.65% | 91.98% | 1,767 | 10 |
| C4.5, no pruning | 78.11% | 81.73% | 379 | 10 |

| Run | runs | CART | C4.5 | RAM GiB (peak) |
|---|---|---|---|---|
| epyc, --simple, one thread | 1 | 14,124 s | 14,240 s | 118.0 |
| dell, --simple, one thread | 1 | 12,414 s | 12,712 s | 118.0 |
| epyc, --serial, one thread | 1 | 878 s | 1,130 s | 346.1 |
| dell, --serial, one thread | 1 | 1,295 s | 1,825 s | 346.1 |
| dell, parallel, one socket | 3 | 58.7 s ± 212 ms | 77.2 s ± 53.5 ms | 345.2 |
| epyc, parallel, both sockets | 3 | 21.8 s ± 509 ms | 28.8 s ± 195 ms | 347.5 |
| dell, parallel, both sockets | 3 | 32.5 s ± 301 ms | 44.3 s ± 197 ms | 347.5 |
