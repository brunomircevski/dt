#!/usr/bin/env python3
"""Bar chart of one benchmark run: training time per tool, single-thread and
multi-thread cases in separate panels, tree size, depth and test accuracy on
each bar. Needs matplotlib (not in requirements.txt).

  bench/.venv/bin/python bench/chart.py bench/results/<run-id> [--out chart.png]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import benchlib as B  # noqa: E402

COLORS = {"tree": "#2b6cb0", "sklearn": "#dd6b20", "rpart": "#38a169", "j48": "#805ad5",
          "yadt": "#c53030"}
ALGORITHM = {"cart_alpha": "CART (fixed α)", "cart_full": "CART (unpruned)",
             "cart_depth12": "CART (depth 12)",
             "c45": "C4.5 (CF 0.25)", "c45_unpruned": "C4.5 (unpruned)"}

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--out")
args = parser.parse_args()
run = Path(args.run)
rows = [json.loads(line) for line in open(run / "results.jsonl")]
check = {(r["protocol"], r["impl"], r["threads"]): r for r in rows
         if r["kind"] == "check" and r["status"] == "ok"}
times = defaultdict(list)
for r in rows:
    if r["kind"] == "time" and r["status"] == "ok":
        times[(r["protocol"], r["impl"], r["threads"])].append(r["train_seconds"])
dataset = rows[0]["dataset"]
meta = B.meta(dataset) if (B.DATA / dataset / "meta.json").exists() else None

panels = {"single": [k for k in times if k[2] == 1], "multi": [k for k in times if k[2] > 1]}
protocols = [p for p in B.PROTOCOLS if any(k[0] == p for k in times)]
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True,
                         gridspec_kw={"width_ratios": [max(len(v), 1) for v in panels.values()]})
for ax, (name, keys) in zip(axes, panels.items()):
    keys = sorted(keys, key=lambda k: (protocols.index(k[0]), list(B.IMPLS).index(k[1])))
    x, labels, previous = 0.0, [], None
    for key in keys:
        protocol, impl, threads = key
        if previous is not None and protocol != previous:
            x += 0.6
        previous = protocol
        seconds = statistics.median(times[key])
        info = check.get(key, {})
        ax.bar(x, seconds, color=COLORS[impl], width=0.8)
        stats = (f"\nacc {info['test_accuracy'] * 100:.2f}%\n{info['nodes']:,} nodes\n"
                 f"depth {info['depth']}") if info else ""
        ax.text(x, seconds * 1.1, f"$\\bf{{{seconds:.3g}\\ s}}${stats}", ha="center",
                va="bottom", fontsize=8.5)
        labels.append((x, f"{B.IMPLS[impl]['label']}\n{ALGORITHM.get(protocol, protocol)}"))
        x += 1
    ax.set_yscale("log")
    ax.set_xticks([p for p, _ in labels], [t for _, t in labels], fontsize=8)
    ax.set_ylabel("training time, seconds (log scale)")
    threads = sorted({k[2] for k in keys})
    ax.set_title("Single thread" if name == "single"
                 else f"Multi-thread ({', '.join(map(str, threads))} threads)")
    ax.grid(axis="y", which="both", alpha=0.3)

rows_text = f"{meta['n_train']:,} train / {meta['n_test']:,} test rows" if meta else ""
every = [t for v in times.values() for t in v]
axes[0].set_ylim(min(every) / 3, max(every) * 12)
axes[1].tick_params(labelleft=True)
fig.suptitle(f"{dataset}: {rows_text}. Above bars: training time, test accuracy, nodes, depth.",
             fontsize=11)
fig.tight_layout()
out = Path(args.out) if args.out else run / "chart.png"
fig.savefig(out, dpi=130)
print(out)
