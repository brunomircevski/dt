"""The fastest --handoff-rows of a sweep (hybrid_vs_gpu_only.sh, handoff_sweep.sh).

    pick_handoff.py <sweep folder> <node> <dataset> <value>...

reads <folder>/cuda-<node>-<dataset>-mr<value>/results.jsonl for each value
("default" = ./tree's own), writes <folder>/<node>-<dataset>.json and prints
the value taken: the smallest mean, over the protocols, of its training time
over that protocol's fastest. "default" is measured for reference only.
"""

import json
import statistics
import sys
from pathlib import Path

folder, node, dataset, values = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
times = {}  # value -> {protocol: mean training seconds}
for value in values:
    results = folder / f"cuda-{node}-{dataset}-mr{value}" / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    by = {}
    for row in rows:
        if row.get("kind") == "run" and row.get("status") == "ok":
            by.setdefault(row["protocol"], []).append(row["train_seconds"])
    if by:
        times[value] = {p: statistics.mean(v) for p, v in by.items()}
protocols = set().union(*times.values()) if times else set()
times = {v: t for v, t in times.items() if set(t) == protocols}  # every protocol measured
numeric = [v for v in times if v != "default"]
if not numeric:
    sys.exit(f"no complete sweep value for {node} {dataset}")
fastest = {p: min(t[p] for t in times.values()) for p in protocols}
score = {v: statistics.mean(t[p] / fastest[p] for p in t) for v, t in times.items()}
best = min(numeric, key=score.get)
(folder / f"{node}-{dataset}.json").write_text(json.dumps(
    {"min_rows": int(best), "times": times, "score": score}, indent=1) + "\n")
print(best)
