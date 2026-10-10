## cpu-legion-susy-scaling

**cpu-legion-susy-scaling**: 13th Gen Intel(R) Core(TM) i7-13650HX, 20 logical CPUs (P-cores 0-11, E-cores 12-19), 31 GiB, Linux 7.2.9-1-cachyos #1 SMP PREEMPT_DYNAMIC Sat, 03 Oct 2026 18:34:55 +0000 x86_64, governor performance, turbo on, ./tree e809e97f6bb0, CPUs in order of use 2,4,6,8,10,0,12-19,3,5,7,9,11,1, 1 run(s) per case, warm-up 50000 rows

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 8 MiB

### susy (4,500,000 train / 500,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.05 s | 8.05 s–8.05 s | 1.00× | 1.19 GiB | 1.18 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.44 s | 4.44 s–4.44 s | 1.00× | 1.72 GiB | 1.71 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.47 s | 2.47 s–2.47 s | 1.00× | 1.72 GiB | 1.71 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.53 s | 1.53 s–1.53 s | 1.00× | 1.72 GiB | 1.71 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.21 s | 1.21 s–1.21 s | 1.00× | 1.72 GiB | 1.71 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.08 s | 1.08 s–1.08 s | 1.00× | 1.72 GiB | 1.72 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit** (`cart_alpha_nodepth`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.03 s | 1.03 s–1.03 s | 1.00× | 1.72 GiB | 1.72 GiB | 723 | 362 | 19 | 79.68% | 79.65% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 6.88 s | 6.88 s–6.88 s | 1.00× | 1.43 GiB | 1.43 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 4.28 s | 4.28 s–4.28 s | 1.00× | 1.97 GiB | 1.97 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 2.35 s | 2.35 s–2.35 s | 1.00× | 1.98 GiB | 1.97 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.47 s | 1.47 s–1.47 s | 1.00× | 1.98 GiB | 1.97 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.25 s | 1.25 s–1.25 s | 1.00× | 1.98 GiB | 1.97 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.07 s | 1.07 s–1.07 s | 1.00× | 1.98 GiB | 1.97 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 1.04 s | 1.04 s–1.04 s | 1.00× | 1.98 GiB | 1.98 GiB | 15,113 | 7,557 | 38 | 80.67% | 79.74% |

Test accuracy is measured on 500,000 rows: its 95% interval is ±0.11 percentage points at most.
