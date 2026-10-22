# Profiling: where CART and C4.5 spend their growing time

**Status: running** (submitted 2026-10-22). The earlier runs, with pruning (CART at α = 1e-5, C4.5 CF 0.25) and C4.5 unpruned, were removed. They are in commit 08c4235.

The new runs grow trees to depth 10 without pruning:
- **CART** = `cart_depth10`: `--cart --no-prune -d 10`.
- **C4.5** = `c45_depth10_unpruned`: `--c45 --no-prune -d 10`. Only C4.5's tidy-up after growing runs; the chart counts it as other.
- **Repetition:** 10 runs per case, profiled and plain (`plain/`). The chart uses the median profiled run.
- **Machines:** epyc1 (parallel), hopper (CUDA hybrid and `--gpu-only`).
- **Datasets:**
  - chess 10M, 50M, 100M, 500M, 1000M and 2000M
  - HIGGS
  - Airline
  - ECBDL'14 15M

```bash
bench/cluster/profile.sh -m 10 -d <dataset> -p cart_depth10,c45_depth10_unpruned -o cluster-profiling \
    cpu-epyc-<dataset> cuda-hopper-<dataset> gpuonly-hopper-<dataset>
bench/.venv/bin/python bench/profile_share_chart.py bench/results/cluster-profiling
```
