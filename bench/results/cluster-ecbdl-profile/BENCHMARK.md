# Where the training time goes: ECBDL'14 15M sample, profiled on the mordor3 cluster

`./tree --profile` on ECBDL'14's 15M-row sample (631 features), one run per case, with and without `--profile`:

- [**gpu-plus-cpu/**](gpu-plus-cpu/BENCHMARK.md) (2026-10-19): CUDA on hopper (H100) and ampere (A100) with all CPU threads, whose hand-offs grow the small subtrees on the CPU, and parallel on epyc1. The H100 grows the trees 2.1–2.5× faster than epyc but prunes C4.5 on its host CPUs; the CPU threads of a GPU node are idle 39–94% of the time.
- [**gpu-only/**](gpu-only/BENCHMARK.md) (2026-10-20): `--cuda --gpu-only --threads 1` on hopper and ampere, every node on the GPU and one CPU thread, against the CUDA runs above. The same trees; unpruned CART takes 2.45× (H100) and 1.45× (A100) the time, C4.5 7.4× and 3.5×, nearly all of it the one-thread prune. Unpruned CART needs 36.5 GiB of RAM instead of 106.7 GiB.
