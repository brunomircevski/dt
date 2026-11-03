# hybrid-vs-gpu-only: ./tree's CUDA hybrid against GPU only, by depth

![ampere](chart-ampere.png)

hopper: queued (job in `hopper.log`), chart `chart-hopper.png` once it ran.

**What.** ./tree `--cuda` (hybrid: a node with fewer rows than
`--gpu-min-rows` is copied to the host and its subtree grown by the CPU
threads while the GPU goes on with the next level) against `--cuda --gpu-only`
(every node, down to the leaves, on the GPU). CART and C4.5 without pruning,
grown to depth 5, 10, 15, 20, 25, 30 and without a depth limit, on two
datasets: airline (109.3M rows × 13 features), where the gap is the largest of
the datasets tried, and MNIST8M (8.1M × 784), where it is the smallest that
both algorithms agree on. 3 runs a case. The trees are the same in both modes
(node counts under each group).

**Where.** ampere: A100 80 GB PCIe (GPU 0), the 24 CPUs of its socket and
`numactl --membind=0` (job.sh's choice: the GPU's NUMA node), the whole node
(`--exclusive`). ./tree fef9ac0, 2026-11-03.

## Results (ampere)

Mean training time of 3 runs (presort, build; not reading the data); the
spread is under 3% in every case. GPU only / hybrid in brackets.

| Depth | airline CART | airline C4.5 | MNIST8M CART | MNIST8M C4.5 |
|---|---|---|---|---|
| 5 | 0.55 s / 0.55 s (1.00×) | 0.56 / 0.55 s (0.98×) | 3.26 / 3.22 s (0.99×) | 3.33 / 3.30 s (0.99×) |
| 10 | 0.80 / 0.80 s (0.99×) | 0.82 / 0.81 s (0.99×) | 5.47 / 5.34 s (0.98×) | 5.41 / 5.53 s (1.02×) |
| 15 | 1.08 / 1.08 s (0.99×) | 1.11 / 1.11 s (1.00×) | 7.86 / 9.07 s (1.15×) | 7.73 / 8.25 s (1.07×) |
| 20 | 1.56 / 1.78 s (1.14×) | 1.49 / 1.67 s (1.12×) | 9.86 / 15.0 s (1.52×) | 9.72 / 11.9 s (1.23×) |
| 25 | 2.14 / 4.85 s (2.26×) | 2.01 / 3.26 s (1.62×) | 11.6 / 18.7 s (1.61×) | 11.4 / 15.9 s (1.39×) |
| 30 | 3.23 / 11.9 s (3.69×) | 2.51 / 6.75 s (2.69×) | 12.6 / 20.9 s (1.66×) | 12.9 / 19.5 s (1.52×) |
| none | 4.49 / 24.8 s (5.53×) | 4.15 / 18.4 s (4.42×) | 18.8 / 30.1 s (1.61×) | 23.4 / 35.8 s (1.53×) |

Times are hybrid / GPU only.

- **Up to depth 15 the modes are equal** (0.98–1.02×, MNIST8M's depth 15 aside):
  few nodes get below `--gpu-min-rows`, so the hybrid hands almost nothing to
  the CPU, and both run the same GPU levels.
- **Deeper, the hybrid wins, and by how much depends on how many small nodes
  the tree has.** airline's unlimited CART tree has 44.7M nodes, nearly all
  of them small: GPU only still pays a whole level's kernels and
  synchronisation for every level of nodes of a few rows, where the hybrid's
  CPU threads grow those subtrees in one go. MNIST8M's trees have 0.2–0.3M
  nodes, so the GPU-only tail is short and the gap stays at about 1.5–1.7×
  from depth 20 on.
- **The gap is all in the build.** GPU setup is the same in both modes
  (airline 0.32–0.33 s, MNIST8M 1.26–1.28 s at no depth limit): neither
  dataset has many distinct values for C4.5's presort, so the hybrid's
  background zeroing of its host arena costs nothing here (unlike sorel_4m's
  C4.5). Build at no depth limit: airline CART 4.17 s hybrid, 24.5 s GPU
  only; MNIST8M CART 17.5 s, 28.9 s.

## --gpu-min-rows: the sweep

Before the depth runs, the hybrid at no depth limit, CART and C4.5, at
./tree's default (CART rows/8192 clamped to 1,024–2,048, C4.5 rows/4096
clamped to 1,024–4,096) and at fixed values, interleaved (`tune.sh`), 3 runs a
value. The value taken has the smallest mean, over CART and C4.5, of its time
over that algorithm's fastest. Data: `sweep/ampere-*.csv`, summary
`sweep/ampere-<dataset>.json`.

| airline | default | 128 | 256 | **512** | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CART (s) | 4.60 | 5.45 | 4.58 | **4.48** | 4.50 | 4.59 | 4.76 | 5.11 | 5.58 | 6.06 | 7.52 |
| C4.5 (s) | 4.51 | 5.60 | 4.40 | **4.16** | 4.16 | 4.28 | 4.51 | 4.90 | 5.49 | 6.15 | 8.06 |

| MNIST8M | default | 512 | 1024 | 2048 | 4096 | **8192** | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|
| CART (s) | 21.4 | 22.3 | 21.2 | 20.5 | 20.0 | **19.0** | 20.5 | 27.0 |
| C4.5 (s) | 25.3 | 27.7 | 26.3 | 25.5 | 24.2 | **23.4** | 22.7 | 26.9 |

airline's defaults are 2,048 (CART) and 4,096 (C4.5), MNIST8M's 1,024 and
1,977: the tuned value is 2–8% faster on airline and 7–12% on MNIST8M. With
784 features a node's CPU work is heavier per row, so MNIST8M wants more of
the GPU (8,192); airline, 13 features, the opposite (512).

**Datasets.** Picked from a screen at no depth limit, one run each
(`sweep/ampere-screen.csv`), GPU only / hybrid, CART and C4.5: airline 5.4× /
4.1×, HIGGS 3.0× / 2.8×, SUSY 3.0× / 1.1×, ECBDL'14 15M 1.5× / 1.3×, MNIST8M
1.44× / 1.40×, Covertype too short to count (0.1 s).

## Commands

ampere ran by hand: the screen and sweep through `bench/cluster/tune.sh`
(`numactl --membind=0 taskset -c 0-23`, `--threads 24`), the depth runs as
four jobs through job.sh:

```bash
RUN_EXTRA="--gpu-min-rows 512" bench/cluster/job.sh hybrid-vs-gpu-only cuda-ampere-airline airline tree_cuda 3 <protocols> all 3600
bench/cluster/job.sh hybrid-vs-gpu-only gpuonly-ampere-airline airline tree_gpu_only 3 <protocols> all 3600
```

(and the same for mnist8m at 8192), protocols `cart_depth{5,10,15,20,25,30}`,
`cart_full_nodepth`, `c45_depth{5,10,15,20,25,30}_unpruned`, `c45_unpruned`.

hopper runs the whole of it (sweep with job.sh, value taken, depth runs) as
one job, `bench/cluster/hybrid_vs_gpu_only.sh`:

```bash
sbatch -J dt-hvg-hopper -p hopper -w hopper --exclusive -t 8:00:00 -o bench/results/hybrid-vs-gpu-only/hopper.log \
    --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty)" bench/cluster/hybrid_vs_gpu_only.sh hopper
```

Chart:

```bash
bench/.venv/bin/python bench/hybrid_chart.py bench/results/hybrid-vs-gpu-only --node ampere --datasets airline,mnist8m
```
