#!/usr/bin/env python3
"""Rerun check: per run folder of bench/results/cluster-*-<suffix>/ and protocol,
the mean and sd of the training time, runs over twice the mean of the other runs,
and the mean of the folder it replaces (cluster-*-fix/, else cluster-*/).

  bench/.venv/bin/python bench/cluster/outliers.py bench/results par2
"""
import collections, csv, glob, statistics as st, sys

def load(path):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["kind"] == "run" and r["status"] == "ok":
            d[r["protocol"]].append(float(r["train_seconds"]))
    return d

root, suffix = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "par2"
for f in sorted(glob.glob(f"{root}/cluster-*-{suffix}/*/runs.csv")):
    run = f.split("/")[-2]; bench = f.split("/")[-3][:-len(suffix) - 1]
    old = {}
    for o in (f"{root}/{bench}-fix/{run}/runs.csv", f"{root}/{bench}/{run}/runs.csv"):
        try: old = load(o); break
        except FileNotFoundError: pass
    for p, t in load(f).items():
        out = [x for i, x in enumerate(t) if len(t) > 1 and x > 2 * st.mean(t[:i] + t[i+1:])]
        o = old.get(p)
        print(f"{bench[8:]:8} {run:24} {p:20} n={len(t):2} mean={st.mean(t):8.3f} sd={st.stdev(t) if len(t) > 1 else 0:7.3f}"
              f"  old={st.mean(o) if o else float('nan'):8.3f} (n={len(o) if o else 0}){'  OUTLIER ' + str(out) if out else ''}")
