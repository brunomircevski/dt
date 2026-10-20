# Where the training time goes with `--gpu-only`: ECBDL'14 15M sample

**Status:** one profiled run per case on 2026-10-20, commit 10d125d (GPU only uses every CPU it may since 3ccf4ba), each node alone and reserved whole: ampere 19:22–19:32, hopper 19:26–19:36. Then the same cases without `--profile` (`plain/`, ampere 19:32–19:42, hopper 19:36–19:46). Each ran on GPU 0 and the CPUs of its NUMA node 0 (16 threads on hopper, 24 on ampere), with the memory bound to that node, like the CUDA runs of [cluster-ecbdl](../../cluster-ecbdl/BENCHMARK.md). The CUDA rows for comparison are those of [gpu-plus-cpu](../gpu-plus-cpu/BENCHMARK.md) (8b1ecd1, the same day, the whole node: 32 and 48 CPU threads, GPU 0), the only profiled CUDA runs; the CUDA runs on the GPU's socket alone (not profiled) are within 4% of them except C4.5 (below). Every case built the same tree as there: 1 leaf (pruned CART), 604,225 nodes (unpruned CART), 209,799 nodes (C4.5).
- **Before:** the GPU-only runs had one CPU thread (7868bf4, the afternoon; and 6da527e, the morning, not bound). Their numbers are below each table and in the git history.

`--cuda --gpu-only`: every node is grown on the GPU, with no hand-offs to the CPU and no host arena for them. The CPU threads feed the GPU, then do C4.5's pruning.

```bash
bench/cluster/profile.sh -o cluster-ecbdl-profile/gpu-only gpuonly-hopper-ecbdl_15m gpuonly-ampere-ecbdl_15m
p=bench/results/cluster-ecbdl-profile
bench/.venv/bin/python bench/profile_chart.py $p/gpu-only/gpuonly-hopper-ecbdl_15m $p/gpu-only/gpuonly-ampere-ecbdl_15m \
    $p/gpu-plus-cpu/cuda-hopper-ecbdl_15m $p/gpu-plus-cpu/cuda-ampere-ecbdl_15m --out-dir $p/gpu-only
```

The charts and `profile.md` (every number behind them) are drawn as in [gpu-plus-cpu](../gpu-plus-cpu/BENCHMARK.md), where *What is measured* explains the phases and kinds of work. Each run's `without --profile` time comes from its sibling `plain/`.

## The training thread's phases

![phases of the training thread: GPU only and CUDA on hopper and ampere](chart-phases.png)

| Train total (s) | GPU only | CUDA (whole node) | GPU only / CUDA | GPU only, 1 thread (7868bf4) |
|---|--:|--:|--:|--:|
| CART, α = 1e-5 · hopper | 8.01 | 8.03 | 1.00× | 9.78 |
| CART, α = 1e-5 · ampere | 11.67 | 11.59 | 1.01× | 12.06 |
| CART, no pruning · hopper | 27.18 | 17.24 | **1.58×** | 28.75 |
| CART, no pruning · ampere | 34.88 | 23.75 | **1.47×** | 34.85 |
| C4.5 · hopper | 60.17 | 46.53 | 1.29× | 242.71 |
| C4.5 · ampere | 83.03 | 65.28 | 1.27× | 226.31 |

- **The GPU grows the small nodes itself, and slower.** That is the whole difference in unpruned CART: its GPU levels take 25.9 s on the H100 (15.9 s with hand-offs) and 32.3 s on the A100 (21.2 s). With hand-offs, 49,187 nodes (5.9M rows) go to the CPU; here the GPU sweeps them, three more levels (94 against 91). Search and partition kernels: 20.0 against 14.0 s on the H100, 26.2 against 18.9 s on the A100.
- **C4.5:** growing 35.9 against 25.7 s (H100) and 46.7 against 34.0 s (A100), for the same reason (274 GPU levels against 272, 21,972 nodes and 8.8M rows handed off with CUDA). The pruning takes 20.8 s with 16 threads against 17.5 s with 32 (H100), and 31.2 s with 24 against 26.0 s with 48 (A100); the CUDA runs on the GPU's socket alone prune in 22.2 and 32.6 s, so with the same CPUs C4.5 is 1.22× and 1.20× (see [cluster-ecbdl](../../cluster-ecbdl/BENCHMARK.md)).
- **Pruned CART** (a single leaf) is only the upload and the top levels, which are the same in both.
- **The upload, presort and GPU setup are the same as with CUDA.**
- **Against one CPU thread** (7868bf4): C4.5 242.7 → 60.2 s (H100) and 226.3 → 83.0 s (A100), the prune 181.0 → 20.8 s and 158.8 → 31.2 s, the ordering by leaf 22.8 → 2.1 s and 18.0 → 2.4 s. The upload 2.5 → 0.70 s on hopper (one thread had staged the data into the page-locked buffers alone) and 2.2 → 1.8 s on ampere. Unpruned CART's GPU levels 25.6 → 25.9 s and 31.9 → 32.3 s: the GPU's growing did not wait on the one thread.

## What the CPU threads do

![thread-seconds by kind of work](chart-threads.png)

Without hand-offs the CPU threads only feed the GPU (`feed`, 15–69 thread-seconds) and, for C4.5, prune (361 thread-seconds on hopper, 788 on ampere). The rest is idle: 113–1,136 thread-seconds, 88–93% of the threads' time in the CART cases. With hand-offs the CPU also sweeps and partitions the small nodes (138–254 thread-seconds in unpruned CART).

## What the GPU runs

![GPU busy time by kernel and copy](chart-gpu.png)

No hand-off kernels or copies; device→host is 0.1 GB for C4.5 (44.7 GB with hand-offs). The extra time is in the search and partition kernels and in `copy other` (1.4–1.7 s against 0.2–0.5 s).

## Timelines

![timeline, CART α = 1e-5](timeline-cart_alpha_nodepth.png)
![timeline, CART without pruning](timeline-cart_full_nodepth.png)
![timeline, C4.5](timeline-c45.png)

## Memory

| Peak | GPU only | CUDA |
|---|--:|--:|
| RAM, CART α = 1e-5 | 36.0 GiB | 37.5 GiB |
| RAM, CART, no pruning | 36.5 GiB | 106.7 GiB |
| RAM, C4.5 | 104.7–104.9 GiB | 107.6–107.9 GiB |
| VRAM (hopper and ampere) | 73.3–74.3 GiB | 73.3–73.5 GiB |

- **Without the host arena,** unpruned CART needs 36.5 GiB of RAM instead of 106.7 GiB. C4.5's pruning needs about 105 GiB in both.
- **VRAM:** unpruned CART takes 0.8–0.9 GiB more on the GPU alone (74.3 GiB), for the deeper levels' buffers; the other cases are within 0.3 GiB.

## What profiling costs

The profiled runs were 0.1–0.6% slower than the runs without `--profile` (`profile.md`).

## Notes

- **Reading the CSV** takes most of each process: about 32 s for pruned CART (`wall_seconds` 40–44 s, against 136 s on hopper and 207 s on ampere with one thread), not part of the train total.
- One run per case: the differences under about 2% are within run-to-run noise.
