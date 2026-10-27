"""Is scikit-learn's ccp pruning really this slow? SUSY subsamples, one thread.

For each row count: a plain fit(ccp_alpha=1e-5), the same fit split into
growing (ccp_alpha=0) and _prune_tree() (what fit does last), whether both
give the same tree, and the leaves of a one-pass bottom-up pruning of the
full tree. Run on a compute node (bench/results/cluster-cpu-vs-others-old/BENCHMARK.md):

  python bench/sklearn_prune_check.py bench/data/susy 250000 500000 1000000 2000000
"""
import sys, time
import numpy as np
from sklearn.tree import DecisionTreeClassifier

F, ROWS, ALPHA = 18, 4_500_000, 1e-5
X = np.fromfile(f"{sys.argv[1]}/train.f32", dtype="<f4").reshape(ROWS, F)
y = np.fromfile(f"{sys.argv[1]}/train.y.i32", dtype="<i4")

def bottom_up(t, alpha):
    """Leaves of the smallest subtree minimising R + alpha * leaves, one pass (children have larger ids)."""
    r = t.weighted_n_node_samples * t.impurity / t.weighted_n_node_samples[0]
    left, right = t.children_left, t.children_right
    cost = np.empty(t.node_count); leaves = np.empty(t.node_count, dtype=np.int64)
    for i in range(t.node_count - 1, -1, -1):
        if left[i] == -1:
            cost[i], leaves[i] = r[i] + alpha, 1
            continue
        sub = cost[left[i]] + cost[right[i]]
        if r[i] + alpha <= sub + 1e-12:
            cost[i], leaves[i] = r[i] + alpha, 1
        else:
            cost[i], leaves[i] = sub, leaves[left[i]] + leaves[right[i]]
    return leaves[0]

for n in map(int, sys.argv[2:]):
    Xs, ys = X[:n], y[:n]
    t0 = time.perf_counter(); plain = DecisionTreeClassifier(ccp_alpha=ALPHA, random_state=0).fit(Xs, ys); t_plain = time.perf_counter() - t0
    t0 = time.perf_counter(); m = DecisionTreeClassifier(random_state=0).fit(Xs, ys); t_grow = time.perf_counter() - t0
    full = m.tree_; full_nodes = full.node_count
    t0 = time.perf_counter(); fast_leaves = bottom_up(full, ALPHA); t_fast = time.perf_counter() - t0
    m.set_params(ccp_alpha=ALPHA)
    t0 = time.perf_counter(); m._prune_tree(); t_prune = time.perf_counter() - t0
    same = (plain.tree_.node_count == m.tree_.node_count
            and np.array_equal(plain.tree_.feature, m.tree_.feature)
            and np.array_equal(plain.tree_.threshold, m.tree_.threshold))
    print(f"n={n:>9,}  full tree {full_nodes:>9,} nodes | plain fit(ccp) {t_plain:8.1f} s | split: grow {t_grow:7.1f} s"
          f" + _prune_tree {t_prune:7.1f} s = {t_grow + t_prune:7.1f} s | pruned {m.tree_.node_count:,} nodes,"
          f" {m.get_n_leaves():,} leaves, same as plain fit: {same} | one-pass bottom-up: {fast_leaves:,} leaves in {t_fast:.1f} s (Python loop)",
          flush=True)
