# Benchmarks against GPU libraries: results, what is timed, fairness

Source: [bench/results/cluster-tree-vs-gpu-libs](../bench/results/cluster-tree-vs-gpu-libs/BENCHMARK.md)
(run 2026-10-17; ./tree rows rerun on 051949b on 2026-10-19; scripts in its `scripts/`).

## Setup

- **Machines (mordor3):** hopper = H100 PCIe 80 GB + 2× Xeon Gold 6444Y
  (32 cores); ampere = A100 80 GB PCIe + 2× EPYC 7402 (48 cores). GPU 0, whole
  node via `--exclusive`, CUDA 12.9.
- **Data:** HIGGS 10.5M train rows × 28 real features; airline 114,569,017
  train rows × 13 integer features; 500,000 test rows each.
- **Task:** one unpruned tree, depth ≤ 19, all rows and features, no
  bootstrap or feature sampling.
- **Competitors:** cuML 26.10 `RandomForestClassifier(n_estimators=1,
  bootstrap=False, max_features=1.0, n_bins=128, n_streams=1)`, Gini for
  CART, entropy (`min_samples_leaf=2`) for C4.5; XGBoost 3.4.1
  `device="cuda", tree_method="hist"`, one round of `reg:squarederror` with
  `eta=1, lambda=0` (= the CART Gini split on 256 bins), CART only.
  CudaTree (2013) does not run on CUDA 12; no public GPU C4.5 code exists.

## Headline results (median of 3)

| Case | ./tree H100 | ./tree A100 | cuML H100 | cuML A100 | XGBoost H100 | XGBoost A100 |
|---|--:|--:|--:|--:|--:|--:|
| HIGGS CART | 0.40 s | 0.48 s | 0.94 s | 1.08 s | 5.41 s | 6.02 s |
| HIGGS C4.5 (cuML: entropy) | 0.36 s | 0.44 s | 0.87 s | 1.03 s | – | – |
| airline CART | 1.39 s | 1.98 s | 4.23 s | 4.67 s | 20.40 s | 24.69 s |
| airline C4.5 (cuML: entropy) | 1.30 s | 1.93 s | 4.18 s | 4.59 s | – | – |

| Case | ./tree nodes / test acc. | cuML | XGBoost | Same predictions as ./tree |
|---|---|---|---|---|
| HIGGS CART | 302,341 / 71.74% | 342,393 / 71.70% | 319,733 / 71.64% | cuML 83.1%, XGB 84.0% |
| airline CART | 621,689 / 74.88% | 655,631 / 74.59% | 641,161 / 74.76% | cuML 86.8%, XGB 90.8% |
| HIGGS C4.5 | 84,695 / 68.65% | 292,477 / 71.76% | – | 76.7% |
| airline C4.5 | 190,337 / 73.02% | 555,457 / 74.26% | – | 85.0% |

| Peak VRAM (above idle, incl. context) | ./tree | cuML | XGBoost |
|---|--:|--:|--:|
| HIGGS | 3.3–3.4 GiB | 1.85–1.91 GiB | 2.86–2.90 GiB |
| airline | 13.3 GiB | 9.5–10.0 GiB | 13.0 GiB |

- Speed-up: CART 2.2–2.4× (HIGGS) and 2.4–3.0× (airline) over cuML,
  12–15× over XGBoost; C4.5 2.3–3.2× over cuML's entropy tree.
- The gap grows with the data (airline > HIGGS). ./tree gains more from the
  H100 over the A100 (1.2–1.5×) than the libraries (1.1–1.2×).
- Binned trees agree with the exact tree on only 83–91% of test rows; the
  exact tree is the most accurate CART tree on both datasets (by ≤ 0.3 points).
  Root thresholds differ even on integer data (airline: ./tree 70.5, cuML 70,
  XGBoost < 71).
- Every tree is identical on H100 and A100.

## What exactly is timed

Both sides: float32 training data in host RAM → finished tree. Not timed on
either side: reading the file, prediction, accuracy.

**./tree**: its printed `train total`, a fresh process per run,
`--cuda --threads <all cores> --cart|--c45 --no-prune -d 19`.
`train total = gpu setup + presort + build + prune` (`src/app/trainer.h`):

| Phase | Contents |
|---|---|
| gpu setup | streams, all device allocations, upload of columns + labels via pinned staging buffers |
| presort | build `(value, row\|class)` entries, CUB radix sort per feature (C4.5: distinct values) |
| build | all GPU levels, hand-offs, waiting for CPU threads to finish small subtrees, assembling the host tree |
| prune | C4.5 only: collapse of useless splits and thresholds set to training values (part of C4.5 even with `--no-prune`) |

Not in ./tree's total: CSV parsing; CUDA context creation and the 4×32 MB
pinned staging buffers (both made on a background thread during the CSV load,
`src/app/main.cpp`); evaluation; `--dump`.

Phase medians (s):

| Case | gpu setup | presort | build | total |
|---|--:|--:|--:|--:|
| H100 HIGGS CART | 0.090 | 0.018 | 0.280 | 0.387 |
| H100 HIGGS C4.5 | 0.099 | 0.027 | 0.233 | 0.361 |
| A100 HIGGS CART | 0.096 | 0.023 | 0.378 | 0.499 |
| A100 HIGGS C4.5 | 0.097 | 0.032 | 0.310 | 0.442 |
| H100 airline CART | 0.216 | 0.075 | 1.091 | 1.385 |
| H100 airline C4.5 | 0.219 | 0.086 | 1.012 | 1.321 |
| A100 airline CART | 0.325 | 0.098 | 1.553 | 1.969 |
| A100 airline C4.5 | 0.320 | 0.110 | 1.452 | 1.888 |

**cuML / XGBoost** (`gpubench.py`): one Python process loads row-major
(C-order) NumPy float32 arrays, runs an **untimed warm-up fit on 50,000 rows**
(imports, CUDA context, JIT, memory pool), then 3 timed fits
(`time.perf_counter`):
- cuML: `fit(X, y)`: input conversion, host→device copy, quantile binning,
  growing.
- XGBoost: `QuantileDMatrix(X, y)` (copy + quantile sketch) + `xgb.train(...,
  num_boost_round=1)`.
- `predict` and the treelite / JSON export are not timed.

VRAM: peak `nvidia-smi memory.used` above idle, sampled every 20 ms; for the
libraries over the whole process (warm-up + 3 fits).

## Fairness

**Matched:** same rows, features, depth limit, no sampling, no pruning, same
GPUs and exclusive nodes, both start from host RAM, both pay their upload and
preprocessing (presort vs binning), medians of 3.

**Asymmetries favouring ./tree:**
1. **CPU threads** (biggest): ./tree uses all 32/48 cores for the small
   subtrees, overlapped with the GPU; cuML runs on the GPU with ~one host
   thread. XGBoost gets `nthread = all cores` but uses them mainly in the
   quantile sketch.
2. **Input layout:** ./tree's host data is already column-major after loading
   (`values[feature * rows + row]`); the libraries get row-major NumPy and any
   transpose happens inside the timed `fit`. Not measured.
3. **Upload path:** ./tree uploads through pinned staging buffers (~55 GB/s
   on hopper); cuML copies from pageable memory (typically 10–25 GB/s). For
   airline's 6 GB that may be ~0.25–0.6 s of cuML's 4.2 s (estimate, not
   measured). A legitimate engineering win, but should be shown separately.
4. Pinned staging buffers allocated during the untimed CSV load (small).

**Asymmetries favouring the libraries:**
1. **Warm process:** their timed fits follow a warm-up, with an already grown
   memory pool (their first full-size fit was 0.9–4 s slower; the median drops
   it). ./tree is a cold process every run and pays its allocations in
   `gpu setup`.
2. CUDA context: hidden in the CSV load for ./tree, in the warm-up for the
   libraries: even.

**Not a timing issue but a comparison issue:** C4.5 vs cuML entropy are
different algorithms; lead with CART.

## What reviewers will ask

- "Is it a GPU algorithm?" → report GPU vs CPU share of rows/nodes and a
  `--threads` / `--gpu-min-rows` sweep.
- "Single tree only?" → cuML is designed for forests; state that the target is
  one exact tree (and the CPU pruning workflows that need it).
- "Where does the time go?" → per-kernel breakdown and achieved bandwidth.
- "Is the comparison end-to-end fair?" → the fairness variants below.

## Experiments still missing

1. **Clean ablation** on one commit and machine: `--gpu-sweep` (one exact pass
   vs float-filter two-pass, also on the RTX 5070 laptop), `--gpu-min-rows`
   sweep, CPU thread sweep; compile-time toggles would be needed for the
   bitmap, fused next-level histograms, `tileBound`, async hand-off ring.
2. **Per-kernel / per-level profile**: `./tree --cuda --gpu-profile` (it
   serializes kernels) plus `nsys` (`make NVTX=1`); achieved bandwidth of
   `partitionKernel` and `tileExact` vs peak (roofline-style statement).
3. **Fairness variants**: libraries with data already on the GPU (cupy,
   column-major) and with `np.asfortranarray` host input; ./tree build-only
   time (`train total − gpu setup`, already in `results.jsonl`) next to them;
   ./tree with `--threads 1/4/8/all`.
4. Optionally: scaling in rows (airline subsamples 1M…115M) for ./tree vs cuML.
5. An `nsys` profile of cuML, to back the explanation of where its time goes.
