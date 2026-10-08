## susy500k-20261008-v2

**susy500k-20261008-v2**: Intel(R) Core(TM) i7-14700KF, 28 logical CPUs (P-cores 0-15, E-cores 16-27), 31 GiB, Linux 7.2.9-arch1-1 #1 SMP PREEMPT_DYNAMIC Sat, 03 Oct 2026 11:43:28 +0000 x86_64, governor powersave, ./tree 656424d4f501-dirty, pinned to CPUs 0-27

Runtime footprint (_baseline dataset, peak RSS / anon): 

### susy_500k (500,000 train / 100,000 test rows, 18 features, 2 classes)

**CART, one tree pruned at a fixed alpha (cost-complexity)** (`cart_alpha`)

| Implementation | Threads | Train time (median) | min–max | vs ./tree 1 thr | Peak RSS | Peak anon | RSS − footprint | Nodes | Leaves | Depth | Test acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ./tree | 1 | 843 ms | 843 ms–843 ms | 1.00× |  |  |  | 3,269 | 1,635 | 30 | 79.26% |
| ./tree | 28 | 126 ms | 126 ms–126 ms | 0.15× |  |  |  | 3,269 | 1,635 | 30 | 79.26% |
| scikit-learn | 1 | 17 s | 17 s–17 s | 20.22× |  |  |  | 3,473 | 1,737 | 25 | 78.94% |
| rpart | 1 | 9.54 s | 9.54 s–9.54 s | 11.32× |  |  |  | 2,863 | 1,432 | 29 | 79.32% |

**C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows** (`c45`)

| Implementation | Threads | Train time (median) | min–max | vs ./tree 1 thr | Peak RSS | Peak anon | RSS − footprint | Nodes | Leaves | Depth | Test acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ./tree | 1 | 639 ms | 639 ms–639 ms | 1.00× |  |  |  | 3,579 | 1,790 | 27 | 79.26% |
| ./tree | 28 | 151 ms | 151 ms–151 ms | 0.24× |  |  |  | 3,579 | 1,790 | 27 | 79.26% |
| Weka J48 | 1 | 25.9 s | 25.9 s–25.9 s | 40.51× |  |  |  | 3,635 | 1,818 | 27 | 79.27% |
| YaDT | 1 | 10.3 s | 10.3 s–10.3 s | 16.15× |  |  |  | 3,533 | 1,767 | 27 | 79.29% |
| YaDT | 28 | 3 s | 3 s–3 s | 4.69× |  |  |  | 3,533 | 1,767 | 27 | 79.29% |
