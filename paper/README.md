# Paper notes: exact CART / C4.5 decision trees on CUDA

Notes for a future paper on `./tree --cuda` (`src/build/gpu_grower.cu`), collected
2026-10-18 from the code, [docs/CUDA.md](../docs/CUDA.md),
[docs/PERFORMANCE.md](../docs/PERFORMANCE.md) and
[bench/results/cluster-gpu-vs-others](../bench/results/cluster-gpu-vs-others/BENCHMARK.md).

| File | Content |
|---|---|
| [pseudocode.md](pseudocode.md) | Algorithms 1–4: level loop with CPU hand-off, warp-ballot sweep, mixed-precision exact filter, single-pass partition |
| [optimizations.md](optimizations.md) | The core idea, why it beats histogram libraries, every optimization with its measured effect |
| [benchmarks.md](benchmarks.md) | Headline results against cuML / XGBoost, exactly what is timed on each side, fairness, what reviewers will ask, experiments still missing |

## The claim in one sentence

`./tree` builds an **exact** (every distinct value tested, byte-identical to the
CPU reference) CART or C4.5 tree on one GPU **2.2–3.2× faster than cuML** and
**12–15× faster than XGBoost**, which split only on 128/256 histogram bins,
with equal or slightly better CART test accuracy.

## Things to get right in the paper

- On the **H100 and A100 the mixed-precision filter is not used**: the grower
  reads the FP32:FP64 ratio and, at ≤ 4:1, runs one exact double-precision
  sweep (`gpu_grower.cu`, `onePass_`). The speed-up over cuML on those GPUs
  comes from the data layout, the partition, the two-class tile bound and the
  CPU/GPU hybrid. The float filter is what makes consumer GPUs (64:1, e.g. the
  RTX 5070 laptop) work.
- ./tree is a **CPU+GPU hybrid**: nodes below `--gpu-min-rows` (~2–4K rows)
  are finished by all CPU cores (32 on hopper, 48 on ampere) while the GPU
  continues. cuML runs on the GPU with essentially one host thread. Report the
  GPU/CPU share and a thread sweep.
- **CART is the clean comparison.** ./tree's C4.5 (gain ratio, MDL thresholds)
  against cuML's entropy tree compares different trees (airline: 190K vs 555K
  nodes, 73.0% vs 74.3% test accuracy at depth 19). No public GPU C4.5 exists.
- **Memory cost of exactness:** ./tree uses 1.3–1.8× cuML's VRAM.
- The existing optimization numbers come from different commits and machines;
  a clean same-commit ablation and a per-kernel profile are still to do (see
  [benchmarks.md](benchmarks.md#experiments-still-missing)). Only the
  2026-10-28 changes (smaller-side bitmap, split column by position,
  upload/presort overlap, GPU root counts) were measured on one commit
  ([optimizations.md](optimizations.md#same-commit-measurement-of-3-10b-and-10c-2026-10-27));
  the benchmark tables predate them and must be rerun.
