## cpu-epyc-higgs

**cpu-epyc-higgs**: AMD EPYC 9255 24-Core Processor, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64, governor performance, turbo on, ./tree 35cf31543002-dirty, CPUs in order of use 0-47, 3 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 7 MiB

### higgs (10,500,000 train / 500,000 test rows, 28 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 15.3 s | 15.3 s–15.3 s | 1.00× | 3.66 GiB | 3.65 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 9.48 s | 9.47 s–9.48 s | 1.00× | 5.69 GiB | 5.68 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.87 s | 4.86 s–4.88 s | 1.00× | 5.69 GiB | 5.68 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3 s | 2.98 s–3.02 s | 1.00× | 5.69 GiB | 5.68 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.2 s | 2.19 s–2.21 s | 1.00× | 5.69 GiB | 5.68 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.85 s | 1.83 s–1.87 s | 1.00× | 5.69 GiB | 5.68 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.61 s | 1.61 s–1.62 s | 1.00× | 5.69 GiB | 5.69 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.46 s | 1.44 s–1.47 s | 1.00× | 5.69 GiB | 5.69 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.41 s | 1.39 s–1.43 s | 1.00× | 5.70 GiB | 5.69 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.41 s | 1.37 s–1.41 s | 1.00× | 5.70 GiB | 5.69 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.29 s | 1.24 s–1.38 s | 1.00× | 5.70 GiB | 5.69 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.25 s | 1.22 s–1.25 s | 1.00× | 5.70 GiB | 5.70 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.15 s | 1.14 s–1.16 s | 1.00× | 5.70 GiB | 5.70 GiB | 1,611 | 806 | 19 | 71.79% |  |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.11 s | 1.11 s–1.14 s | 1.00× | 5.71 GiB | 5.70 GiB | 1,611 | 806 | 19 | 71.79% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 31.1 s | 31.1 s–31.2 s | 1.00× | 5.02 GiB | 5.02 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 19.7 s | 19.7 s–19.7 s | 1.00× | 7.09 GiB | 7.09 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 10 s | 9.99 s–10 s | 1.00× | 7.13 GiB | 7.12 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 5.67 s | 5.65 s–5.68 s | 1.00× | 7.14 GiB | 7.13 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.1 s | 4.09 s–4.11 s | 1.00× | 7.14 GiB | 7.13 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 3.33 s | 3.33 s–3.39 s | 1.00× | 7.14 GiB | 7.13 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.9 s | 2.85 s–2.93 s | 1.00× | 7.14 GiB | 7.14 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.55 s | 2.54 s–2.58 s | 1.00× | 7.15 GiB | 7.14 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.4 s | 2.39 s–2.44 s | 1.00× | 7.15 GiB | 7.14 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.28 s | 2.25 s–2.37 s | 1.00× | 7.15 GiB | 7.14 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.13 s | 2.1 s–2.14 s | 1.00× | 7.15 GiB | 7.14 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.02 s | 1.97 s–2.08 s | 1.00× | 7.15 GiB | 7.14 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.91 s | 1.89 s–2.03 s | 1.00× | 7.15 GiB | 7.14 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.77 s | 1.75 s–1.78 s | 1.00× | 7.15 GiB | 7.14 GiB | 1,159,295 | 579,648 | 71 | 87.88% |  |
