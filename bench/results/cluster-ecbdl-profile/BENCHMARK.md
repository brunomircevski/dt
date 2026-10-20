# Where the training time goes: ECBDL'14 15M sample, profiled on the mordor3 cluster

`./tree --profile` on ECBDL'14's 15M-row sample (631 features), one run per case, with and without `--profile`:

- [**gpu-plus-cpu/**](gpu-plus-cpu/BENCHMARK.md) (2026-10-19): CUDA on hopper (H100) and ampere (A100) with all CPU threads, whose hand-offs grow the small subtrees on the CPU, and parallel on epyc1. The H100 grows the trees 2.1–2.5× faster than epyc but prunes C4.5 on its host CPUs; the CPU threads of a GPU node are idle 39–94% of the time.
- [**gpu-only/**](gpu-only/BENCHMARK.md) (2026-10-20): `--cuda --gpu-only` on hopper and ampere, every node on the GPU, with the CPUs of the GPU's socket (16 and 24 threads), against the CUDA runs above. The same trees; pruned CART takes the same time, unpruned CART 1.58× (H100) and 1.47× (A100), C4.5 1.29× and 1.27×: the GPU grows the small nodes that CUDA hands to the CPU, slower. Unpruned CART needs 36.5 GiB of RAM instead of 106.7 GiB. (With one CPU thread, earlier that day, C4.5 took 3.5–5.2× the time, in the prune.)
