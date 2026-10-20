# Where the training time goes with `--gpu-only`: ECBDL'14 15M sample, one CPU thread

**Status:** one profiled run per case on 2026-10-20, commit 7868bf4, each node alone and reserved whole: hopper 15:24–15:41, ampere 15:44–16:05. Then the same cases without `--profile` (`plain/`, hopper 15:41–16:00, ampere 16:05–16:25). The one CPU thread ran on a CPU of the GPU's NUMA node, with its memory there (`profile.sh -n`): hopper's node 1 with GPU 2, ampere's node 0 with GPU 0. The CUDA rows for comparison are those of [gpu-plus-cpu](../gpu-plus-cpu/BENCHMARK.md) (8b1ecd1, the same day, 32 and 48 CPU threads, GPU 0). Every case built the same tree as there: 1 leaf (pruned CART), 604,225 nodes (unpruned CART), 209,799 nodes (C4.5).
- **Before:** the first runs (6da527e, the morning of the same day) were not bound and used GPU 0; on hopper, whose NUMA node 0 is nearly full with another user's data, they were much slower (below). 7868bf4 also keeps the staged copies from starting more copy threads than the one CPU.

`--cuda --gpu-only --threads 1`: every node is grown on the GPU, with no hand-offs to the CPU and no host arena for them, and the CPU side is a single thread. That thread feeds the GPU, then does C4.5's pruning alone.

```bash
bench/cluster/profile.sh -n 1 -o cluster-ecbdl-profile/gpu-only gpuonly-hopper-ecbdl_15m   # GPU 2, NUMA node 1
bench/cluster/profile.sh -n 0 -o cluster-ecbdl-profile/gpu-only gpuonly-ampere-ecbdl_15m   # GPU 0, NUMA node 0
p=bench/results/cluster-ecbdl-profile
bench/.venv/bin/python bench/profile_chart.py $p/gpu-only/gpuonly-hopper-ecbdl_15m $p/gpu-only/gpuonly-ampere-ecbdl_15m \
    $p/gpu-plus-cpu/cuda-hopper-ecbdl_15m $p/gpu-plus-cpu/cuda-ampere-ecbdl_15m --out-dir $p/gpu-only
```

The charts and `profile.md` (every number behind them) are drawn as in [gpu-plus-cpu](../gpu-plus-cpu/BENCHMARK.md), where *What is measured* explains the phases and kinds of work. Each run's `without --profile` time comes from its sibling `plain/`.

## The training thread's phases

![phases of the training thread: GPU only and CUDA on hopper and ampere](chart-phases.png)

| Train total (s) | GPU only | CUDA | GPU only / CUDA |
|---|--:|--:|--:|
| CART, α = 1e-5 · hopper | 9.78 | 8.03 | 1.22× |
| CART, α = 1e-5 · ampere | 12.06 | 11.59 | 1.04× |
| CART, no pruning · hopper | 28.75 | 17.24 | **1.67×** |
| CART, no pruning · ampere | 34.85 | 23.75 | 1.47× |
| C4.5 · hopper | 242.71 | 46.53 | **5.2×** |
| C4.5 · ampere | 226.31 | 65.28 | 3.5× |

- **C4.5's pruning on one thread is the big cost.** The prune takes 181.0 s on hopper (17.5 s with 32 threads, 10.4×) and 158.8 s on ampere (26.0 s with 48 threads, 6.1×). Ordering the rows by leaf for it takes 22.8 and 18.0 s (1.8 and 2.5 s). On one thread, ampere's EPYC 7402 prunes 1.14× faster than hopper's Xeon 6444Y; C4.5 is 1.07× faster on ampere overall.
- **The GPU grows the small nodes itself, and slower.** Unpruned CART's GPU levels take 25.6 s on the H100 (15.9 s with hand-offs) and 31.9 s on the A100 (21.2 s). With hand-offs, 49,187 nodes (5.9M rows) go to the CPU; here the GPU sweeps them, three more levels (94 against 91). Search and partition kernels: 19.9 s against 14.0 s on the H100, 26.2 against 18.9 s on the A100.
- **The upload waits on the one thread.** 2.5 s on hopper against 0.7 s with 32 threads, though the copies keep the GPU busy only 0.69 s in both: one thread stages the data into the page-locked buffers. On ampere 2.2 s against 1.7 s.
- **Pruned CART** (a single leaf) is only the upload and the top levels: 1.22× on hopper (the upload), 1.04× on ampere.
- **The presort and the GPU setup are unchanged.**
- **Against the first, unbound runs** (6da527e): hopper's C4.5 379.5 → 242.7 s (the prune 283.8 → 181.0 s, the ordering by leaf 52.8 → 22.8 s), its unpruned CART 42.5 → 28.8 s (the GPU levels 39.0 → 25.6 s, with the same kernel time: the GPU had waited on the host thread), the upload 2.9 → 2.5 s. ampere changed by less than 1%, apart from its prune (160.1 → 158.8 s). The thread and its memory are now on the GPU's node, where memory is free; before, on hopper, they were wherever the kernel put them.

## What the CPU thread does

![thread-seconds by kind of work](chart-threads.png)

The one thread is never idle: feeding the GPU (`feed`) for the whole build, then C4.5's prune (203.8 s on hopper, 176.8 s on ampere). With hand-offs, 32 or 48 threads spend most of their time idle (238–1,482 thread-seconds).

## What the GPU runs

![GPU busy time by kernel and copy](chart-gpu.png)

No hand-off kernels or copies; device→host is 0.1 GB for C4.5 (44.7 GB with hand-offs). The extra time is in the search and partition kernels and in `copy other` (1.3–1.5 s against 0.2–0.5 s).

## Timelines

![timeline, CART α = 1e-5](timeline-cart_alpha_nodepth.png)
![timeline, CART without pruning](timeline-cart_full_nodepth.png)
![timeline, C4.5](timeline-c45.png)

## Memory

| Peak | GPU only | CUDA |
|---|--:|--:|
| RAM, CART α = 1e-5 | 36.0 GiB | 37.5 GiB |
| RAM, CART, no pruning | 36.5–36.6 GiB | 106.7 GiB |
| RAM, C4.5 | 104.2 GiB | 107.6–107.9 GiB |
| VRAM (hopper and ampere) | 73.3–74.3 GiB | 73.3–73.5 GiB |

- **Without the host arena,** unpruned CART needs 36.5 GiB of RAM instead of 106.7 GiB. C4.5's pruning needs about 104 GiB in both.
- **VRAM:** unpruned CART takes 0.8–0.9 GiB more on the GPU alone (74.3 GiB), for the deeper levels' buffers; the other cases are within 0.2 GiB.
- **ampere's VRAM was not recorded in the first runs:** the nvidia-smi sampler reported a flat 425 MiB, because run.py read the samples from a pipe only at the end, and the full pipe (64 KiB, about 170 s of 10 ms samples) blocked nvidia-smi during ampere's one-thread CSV read. Fixed in ba22356.

## What profiling costs

The profiled runs were within −0.7% and +0.5% of the runs without `--profile` (`profile.md`).

## Notes

- **Reading the CSV on one thread** takes most of each process: 136 s (hopper) and 207 s (ampere) for pruned CART (14–19 s with all threads), not part of the train total.
- One run per case: the differences under about 2% are within run-to-run noise.
