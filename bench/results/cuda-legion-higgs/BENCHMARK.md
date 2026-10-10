# HIGGS on the Legion laptop: ./tree parallel and CUDA

**Status:** run on 2026-10-10 at 12:46 (at the same time as `cpu-pc-higgs`, on the other machine), one run per case, commit abc0a22, AC power, platform profile `max-power`, `performance` governor, load average 0.06. The 8 prepared data files have the same SHA-256 as in `cpu-pc-higgs` (`meta.json` differs only in the source's column and label names), and every case built the same tree.

The laptop half of a laptop-vs-desktop comparison on HIGGS; the desktop half is [`cpu-pc-higgs`](../cpu-pc-higgs/BENCHMARK.md). Both are fresh runs; nothing is copied.

## Run it

```bash
bench/results/cuda-legion-higgs/run.sh          # each case once
bench/results/cuda-legion-higgs/run.sh -m 5     # each case 5 times
```

- **Time:** a few minutes, plus preparing HIGGS once.
- **Before:** the same as [`cuda-legion-susy`](../cuda-legion-susy/BENCHMARK.md): commit the code, AC power, platform profile `max-power`, `performance` governor, nothing else on the GPU.
- **What `run.sh` does:** as in `cuda-legion-susy`, with
  `bench/run.py higgs --protocols cart_alpha,c45 --impls tree,tree_cuda --threads all --cpus 4,0-3,5-19 --reps <-m> --timeout 3600`
- **Chart** (`chart.png`, drawn by `run.sh`): this run with the desktop's `cpu-pc-higgs` as hatched reference, the hardware of both machines in its header, RAM and VRAM as separate bars. It refuses to draw if the data files or trees differ. Alone:
  ```bash
  bench/.venv/bin/python bench/machines_chart.py bench/results/cuda-legion-higgs \
      --reference bench/results/cpu-pc-higgs
  ```
  Run `cpu-pc-higgs` first: the chart needs it.
- `machine.json` got its `ram_modules` on 2026-10-10 after the run (read from udev on the same laptop), as did `cpu-pc-higgs`'s.

## Method

The protocols, data and measurements of [`cpu-pc-higgs`](../cpu-pc-higgs/BENCHMARK.md) (where the HIGGS data and its check are described), on the laptop, with two backends:

| Backend | Command | Threads |
|---|---|---|
| Parallel | `./tree_cpu --parallel --threads 20` | 20 (all) |
| CUDA | `./tree --cuda --threads 20` | GPU, plus 20 CPU threads for the small subtrees |

- **GPU memory:** ./tree keeps two copies of all sorted columns on the GPU: 10.5M rows × 28 features × 8 bytes × 2 ≈ 4.4 GiB, plus the CUDA context; the RTX 5070 Laptop GPU has 8 GB. ./tree checks the free memory before allocating and stops with an error if it does not fit.
- **Memory measured:** peak RSS up to the end of training; for CUDA also the peak GPU memory above idle (`nvidia-smi` every 10 ms).

## Results

![Laptop vs desktop on HIGGS](chart.png)

Training time, one run per case, both machines run fresh:

| | CART | C4.5 |
|---|--:|--:|
| Desktop parallel (i7-14700KF, 28 threads) | 4.45 s | 7.29 s |
| Laptop parallel (i7-13650HX, 20 threads) | 4.60 s | 6.65 s |
| Laptop CUDA (RTX 5070 Laptop + 20 threads) | **2.06 s** | **4.08 s** |

- **Same trees:** CART 1,611 nodes, depth 19, test accuracy 71.59%; C4.5 1,160,123 nodes, depth 71, 70.10%, on both machines and both backends.
- **CUDA:** 2.2× (CART) and 1.6× (C4.5) faster than the laptop's 20 threads; 2.2× and 1.8× faster than the desktop's 28. About the same gain over the CPU as on SUSY (2.1× and 1.7× over the laptop's 20 threads there). C4.5 gains less than CART on both datasets; its trees are much larger (here 1.16 million nodes), and ./tree --cuda grows small subtrees on the CPU.
- **Fits the GPU:** 4.67 GiB of the 8 GB (CART; C4.5 4.66 GiB), CUDA context included. Host peak RSS 3.83 GiB (CART) and 6.06 GiB (C4.5), below the parallel runs.
- **Parallel:** the laptop's 20 threads match the desktop's 28 on CART (4.60 vs 4.45 s) and beat them on C4.5 (6.65 vs 7.29 s). The desktop also needs more memory (6.05 vs 5.40 GiB, 8.28 vs 7.63 GiB): more threads, more per-thread buffers.

### Why the laptop's 20 threads beat the desktop's 28 on C4.5

The desktop's CPU is faster (serial runs are 11–12% faster on it), but the parallel build is limited by memory bandwidth, and the laptop has the faster RAM.

| Measured on 2026-10-10 | Desktop | Laptop |
|---|--:|--:|
| RAM | DDR4-3200, 2 channels (51.2 GB/s peak) | DDR5-4800, 2 channels (76.8 GB/s peak) |
| STREAM-style triad, all threads | 36 GB/s (28 threads) | 53 GB/s (20 threads) |
| Triad, 1 thread | 28.5 GB/s | 28.9 GB/s |

./tree's own phase timings on HIGGS, `--parallel`, 3 runs each (ms; spread ≤ 3%, so the differences are real):

| | Presort | Build | Prune | Train total |
|---|--:|--:|--:|--:|
| C4.5 desktop (28 threads) | 949–955 | 5,035–5,100 | 1,228–1,300 | 7,274–7,289 |
| C4.5 laptop (20 threads) | 637–652 | 4,727–4,794 | 1,301–1,377 | 6,668–6,762 |
| CART desktop (28 threads) | 795–796 | 3,615–3,666 | 9–11 | 4,419–4,471 |
| CART laptop (20 threads) | 517–534 | 4,185–4,226 | 10–11 | 4,727–4,761 |

- **Presort** (sorting every column: streaming through memory) is 1.5× faster on the laptop in both algorithms, the same ratio as the measured bandwidth (53 / 36).
- **Build:** CART's build takes 13% less time on the desktop (more cores, higher clocks); C4.5's takes 6% less on the laptop, so C4.5's build depends more on memory bandwidth than CART's (not broken down further).
- **Prune** (largely serial) takes 6% less time on the desktop.
- **Net:** C4.5 gains 0.3 s (presort) + 0.3 s (build) on the laptop and loses 0.1 s (prune): 0.6 s faster. CART gains 0.3 s on presort but loses 0.55 s on build: 0.3 s slower.

