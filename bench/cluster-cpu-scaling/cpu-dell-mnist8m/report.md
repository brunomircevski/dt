## cpu-dell-mnist8m

**cpu-dell-mnist8m**: INTEL(R) XEON(R) GOLD 6542Y, 48 logical CPUs (P-cores None, E-cores None), 503 GiB, Linux 7.0.0-42-generic #42~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Oct 19 14:43:01 UTC 2 x86_64, governor powersave, turbo on, ./tree 81bf6e6c8979, CPUs in order of use 0-47, 4 run(s) per case, every run a fresh process, no warm-up

Runtime footprint (peak RSS on the 200-row _baseline dataset): ./tree 6 MiB

### mnist8m (8,100,000 train / 10,000 test rows, 784 features, 10 classes)

**CART, grown to depth 10, no pruning** (`cart_depth10`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 203 s | 199 s–205 s | 1.00× | 71.12 GiB | 71.12 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 102 s | 99.5 s–105 s | 1.00× | 71.18 GiB | 71.18 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 55.5 s | 52.3 s–59 s | 1.00× | 71.31 GiB | 71.30 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 29.9 s | 28.2 s–30.7 s | 1.00× | 71.55 GiB | 71.55 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 20.3 s | 19.4 s–21.3 s | 1.00× | 71.80 GiB | 71.79 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 15.8 s | 15.4 s–16.4 s | 1.00× | 72.05 GiB | 72.04 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 12.7 s | 12.5 s–13.1 s | 1.00× | 72.29 GiB | 72.29 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 11.1 s | 10.9 s–11.5 s | 1.00× | 72.54 GiB | 72.53 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 10 s | 9.83 s–10.3 s | 1.00× | 72.79 GiB | 72.78 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 9.12 s | 9.03 s–9.19 s | 1.00× | 73.03 GiB | 73.03 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.52 s | 8.46 s–8.62 s | 1.00× | 73.28 GiB | 73.28 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.14 s | 8.07 s–8.18 s | 1.00× | 73.53 GiB | 73.52 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 7.74 s | 7.69 s–7.81 s | 1.00× | 73.77 GiB | 73.77 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**CART, grown to depth 10, no pruning** (`cart_depth10`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 7.57 s | 7.45 s–7.69 s | 1.00× | 74.02 GiB | 74.01 GiB | 2,047 | 1,024 | 10 | 80.77% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 1 thread

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 204 s | 200 s–208 s | 1.00× | 71.13 GiB | 71.12 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 2 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 109 s | 106 s–111 s | 1.00× | 71.18 GiB | 71.17 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 4 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 59.4 s | 54.8 s–60.5 s | 1.00× | 71.30 GiB | 71.30 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 8 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 31.7 s | 30.5 s–32.3 s | 1.00× | 71.55 GiB | 71.55 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 12 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 21.9 s | 20.7 s–22.1 s | 1.00× | 71.80 GiB | 71.79 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 16 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 17.1 s | 16.7 s–17.5 s | 1.00× | 72.04 GiB | 72.04 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 20 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 13.8 s | 13.7 s–14.2 s | 1.00× | 72.29 GiB | 72.28 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 24 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 12.2 s | 12.1 s–12.5 s | 1.00× | 72.54 GiB | 72.53 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 28 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 11 s | 10.9 s–11.2 s | 1.00× | 72.78 GiB | 72.78 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 32 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 10.3 s | 10.2 s–10.4 s | 1.00× | 73.03 GiB | 73.02 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 36 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 9.63 s | 9.43 s–9.67 s | 1.00× | 73.27 GiB | 73.27 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 40 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 9.12 s | 9.03 s–9.32 s | 1.00× | 73.52 GiB | 73.51 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 44 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.77 s | 8.62 s–8.85 s | 1.00× | 73.76 GiB | 73.76 GiB | 1,453 | 727 | 10 | 70.96% |  |

**C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)** (`c45_depth10_unpruned`), 48 threads

| Implementation | Train time (median) | min–max | × ./tree | Peak RSS | RSS − footprint | Nodes | Leaves | Depth | Train acc. | Test acc. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ./tree | 8.58 s | 8.41 s–9.07 s | 1.00× | 74.01 GiB | 74.00 GiB | 1,453 | 727 | 10 | 70.96% |  |
