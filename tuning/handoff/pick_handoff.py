"""The fastest --handoff-rows of a sweep (handoff_sweep.sh).

    pick_handoff.py <sweep folder> <node> <dataset> <key> <value>...

reads <folder>/cuda-<node>-<dataset>-mr<value>/results.jsonl for each value
("default" = ./tree's own), writes <folder>/<node>-<dataset>.json and prints
the value taken: the smallest mean, over the protocols, of its training time
without the GPU setup (train_seconds - gpu_setup_seconds: presort and build,
what the hand-off acts on) over that protocol's fastest. The setup (allocation
and upload) does not depend on the value, and on the H100 it takes one of two
speeds from run to run (docs/PERFORMANCE.md), enough to decide a close sweep. "default" is measured for reference only. The
value is also stored in tuning/handoff/handoff_rows.json under node, dataset and <key>
("depth10", "nodepth"), where bench/run.py takes it for every --hybrid run.
"""

import fcntl
import json
import statistics
import sys
from pathlib import Path

folder, node, dataset, key, values = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
times = {}  # value -> {protocol: mean training seconds without the GPU setup}
for value in values:
    results = folder / f"cuda-{node}-{dataset}-mr{value}" / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    by = {}
    for row in rows:
        if row.get("kind") == "run" and row.get("status") == "ok":
            by.setdefault(row["protocol"], []).append(
                row["train_seconds"] - row.get("gpu_setup_seconds", 0.0))
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

store = Path(__file__).resolve().parent / "handoff_rows.json"
with open(store.with_suffix(".lock"), "w") as lock:  # both GPU nodes may write at once
    fcntl.flock(lock, fcntl.LOCK_EX)
    data = json.loads(store.read_text()) if store.exists() else {}
    data.setdefault(node, {}).setdefault(dataset, {})[key] = int(best)
    store.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
print(best)
