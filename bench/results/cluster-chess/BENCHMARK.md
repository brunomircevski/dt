# Chess 3×3 on the mordor3 cluster: 1M to 1000M rows on two GPU and three CPU servers, unpruned CART and C4.5

**Status:** three runs per case, all on 2026-10-19 between 17:34 and 17:42, with the code of commit 2a9d64e on the cluster (on the laptop it is 5058ce1, the same change cherry-picked onto e0a9106; ./tree itself is unchanged since 051949b).
- **Each node ran alone** and was used whole (`--exclusive`). The four sizes ran one after another on every node, smallest first.
- **Same trees:** every case built the same tree on every node and backend: 17 nodes (9 leaves), depth 4, 100% training accuracy, at every size. That is the exact tree of a 3×3 board.
- **Settings:** ./tree's defaults. The trees are grown without pruning (`cart_full_nodepth`, `c45_unpruned`), with no depth limit. Only training time and memory are compared; accuracy is ./tree's training accuracy (`run.py --train-accuracy-only`).

How the runs are made, and where the data came from: [bench/cluster/README.md](../../cluster/README.md).

```bash
for d in chess_1m chess_10m chess_100m chess_1b; do
  bench/cluster/submit.sh -m 3 -d $d -p cart_full_nodepth,c45_unpruned -a <prepare job> \
      cuda-hopper-$d cuda-ampere-$d cpu-epyc-$d cpu-dell-$d cpu-dell512-$d
done
```

The charts are drawn as shown in the README: `chart.png` (one `machines_chart.py` chart per size, in a 2×2 grid) and `chart-rows.png` (`rows_chart.py`).

**The dataset** is a 3×3 checkerboard: two features, x and y, uniform in [0, 3] with six decimals. The class (`A`, `B`) alternates over the nine unit squares (`B` on five of them: 55.5% of the rows), with no noise. Four sizes:

| Dataset | Training rows | Test rows (own file) | CSV | Note |
|---|--:|--:|--:|---|
| `chess_1m` | 1,000,000 | 1,000 | 20 MB | the first 1M rows of `chess_10m` |
| `chess_10m` | 10,000,000 | 4,500 | 200 MB | the first 10M rows of `chess_100m` |
| `chess_100m` | 100,000,000 | 4,500 | 2.0 GB | |
| `chess_1b` | 1,000,000,000 | 1,000 | 20.0 GB | `chess_100m` ten times over: 100M distinct rows, each 10 times |

**What it measures.** The tree is tiny (4 levels, 8 splits), so a run is the presort plus a few full sweeps over all rows: time per row at the top levels, without the many small nodes of a deep tree. With two features, the CPU backend has only two columns to share among 48 threads.

| Run | Node | CPU | RAM | GPU (GPU 0 used) | Backend |
|---|---|---|---|---|---|
| `cuda-hopper-*` | hopper | 2× Xeon Gold 6444Y, 32 cores | 1 TB DDR5-4800 | H100 PCIe 80 GB | CUDA (+ 32 CPU threads) |
| `cuda-ampere-*` | ampere | 2× EPYC 7402, 48 cores | 1 TB DDR4-3200 | A100 80 GB PCIe | CUDA (+ 48 CPU threads) |
| `cpu-epyc-*` | epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – | parallel (48 threads) |
| `cpu-dell-*` | dell1 | 2× Xeon Gold 6542Y, 48 cores | 1 TB DDR5-5200 (16 × 64 GB) | – | parallel (48 threads) |
| `cpu-dell512-*` | dell2 | the same as dell1 | 512 GB DDR5-5200 (8 × 64 GB) | – | parallel (48 threads) |

## Time and memory against the rows

![time, RAM and VRAM against the training rows](chart-rows.png)

Training time (`train total`: presort and build, on CUDA also GPU setup and upload; not reading the CSV), median of 3 runs:

| CART, no pruning | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| CUDA · hopper (H100 PCIe + 32 threads) | 53 ms | **78 ms** | **154 ms** | **1.12 s** |
| CUDA · ampere (A100 80GB PCIe + 48 threads) | 41 ms | 95 ms | 197 ms | 1.79 s |
| Parallel · epyc (EPYC 9255, 48 threads) | **22 ms** | 175 ms | 1.15 s | 47.19 s |
| Parallel · dell (Xeon 6542Y, 48 threads) | 70 ms | 265 ms | 2.43 s | 34.20 s |
| Parallel · dell512 (the same, half the memory channels) | 57 ms | 249 ms | 2.75 s | 36.00 s |

| C4.5, no pruning | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| CUDA · hopper | 53 ms | **68 ms** | **159 ms** | **1.14 s** |
| CUDA · ampere | 40 ms | 87 ms | 200 ms | 1.79 s |
| Parallel · epyc | **25 ms** | 192 ms | 1.35 s | 40.98 s |
| Parallel · dell | 73 ms | 357 ms | 3.47 s | 43.33 s |
| Parallel · dell512 | 64 ms | 289 ms | 3.18 s | 48.32 s |

Peak memory (the same for CART and C4.5 within 0.01 GiB):

| | 1M | 10M | 100M | 1000M |
|---|--:|--:|--:|--:|
| RAM, CPU servers | 0.06 GiB | 0.48 GiB | 4.68 GiB | 46.70 GiB |
| RAM, CUDA (hopper / ampere) | 0.41 / 0.40 GiB | 0.63 / 0.62 GiB | 2.81 / 2.80 GiB | 24.62 / 24.61 GiB |
| VRAM, GPU 0 (hopper / ampere) | 0.53 / 0.49 GiB | 0.89 / 0.85 GiB | 3.44 / 3.41 GiB | 25.46 / 25.44 GiB |

- **The GPUs pull away with the rows.** At 1M rows epyc is the fastest (22 ms): a GPU run's fixed costs (setup, transfers, kernel launches) are about 40–50 ms. From 10M rows on the GPUs lead: the H100 is 2.2–2.8× faster than epyc at 10M, 7.4–8.5× at 100M, and **30× (CART, against dell) to 36× (C4.5, against epyc) at 1000M**. The A100 is 19–23× faster than the best CPU at 1000M.
- **The GPU time grows slower than the rows**, 7× from 100M to 1000M on the H100 (9× on the A100): even 100M rows do not keep the GPU busy, 1000M do.
- **The CPU time grows faster than the rows from 100M to 1000M**: 10× the rows, but 41× the time on epyc for CART and 30× for C4.5, 13–15× on dell and dell512. The presort grows 10× as it should (epyc 0.20 → 1.93 s, dell 0.55 → 4.07 s); the build does not (epyc CART 0.95 → 45.3 s, dell 1.86 → 30.4 s). At 10M and 100M epyc is 1.5–2.6× faster than dell, as on the other datasets; at 1000M it is 1.4× slower on CART and only 1.06× faster on C4.5. Not looked into here: two features leave most of the 48 threads without a column to sweep at the top levels, and 1000M is the first size whose columns (8 GB each) are far beyond any cache and the first whose rows repeat (each 10 times).
- **dell512 against dell:** from 19% faster to 13% slower, unlike ECBDL'14, where half the memory channels made dell2 1.35–1.55× slower. With two features the build is not bound by memory bandwidth.
- **Memory is linear in the rows:** 50 bytes per row on the CPU servers (46.7 GiB for 1000M rows, 2 G values: 25 bytes per value) and 27 bytes per row of VRAM (25.5 GiB of the 80 GB). Every size fits on every GPU; the 40 GB A100s of ampere would hold the 1000M set too. A CUDA run needs about half the host RAM of a CPU run (24.6 GiB at 1000M).

## One chart per size

![chess 3x3: 1M, 10M (top), 100M and 1000M (bottom) on five mordor3 nodes](chart.png)

## Notes

- **Not timed:** reading the CSV, and the data check before each job (SHA-256 of every prepared file and every CSV row against the binary files: 113 s for `chess_1b` on ampere, 33 GB over NFS). A whole `chess_1b` process (`wall_seconds`, reading the 20 GB CSV included) took 5.2–6.5 s on the GPU nodes and 37–52 s on the CPU nodes.
- **Hopper's and ampere's 1M and 10M runs** are dominated by fixed costs; their order there says little about the GPUs.
- **The originals** (C4.5 files from mordor3's `~/chess`) were deleted after the conversion; `datasets/chess_*.csv` on mordor3 are the only copies. Their SHA-256 and those of the originals are in [bench/cluster/README.md](../../cluster/README.md) and `bench/cluster/chess-originals.sha256`.
