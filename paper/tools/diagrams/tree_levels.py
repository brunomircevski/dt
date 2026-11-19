#!/usr/bin/env python3
"""Per-level node statistics of a tree written by ./tree --dump (indent = depth, "[n=...]").

  ./tree_cpu --multicore --cart -d 30 --no-accuracy --dump dump.txt ~/datasets/higgs/train.f32
  python3 paper/tools/diagrams/tree_levels.py dump.txt paper/tools/diagrams/data/higgs-cart-d30-levels.json

Per level d: nodes, rows (samples), median and largest node, and the nodes and
rows in nodes of at least 32,768 / 4,096 / 512 samples.
data/higgs-cart-d30-levels.json was made this way on 2026-11-18 (dell-512GB, tree_cpu).
"""
import sys, re, collections, json
cnt = collections.Counter(); rows = collections.Counter(); sizes = collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.search(r'\[n=(\d+)\]', line)
    if not m:
        continue
    d = (len(line) - len(line.lstrip(' '))) // 2; n = int(m.group(1))
    cnt[d] += 1; rows[d] += n; sizes[d].append(n)
out = []
for d in sorted(cnt):
    s = sorted(sizes[d])
    out.append(dict(d=d, nodes=cnt[d], rows=rows[d], med=s[len(s) // 2], mx=s[-1],
        ge32k=sum(x >= 32768 for x in s), ge4k=sum(x >= 4096 for x in s), ge512=sum(x >= 512 for x in s),
        rows_ge32k=sum(x for x in s if x >= 32768), rows_ge4k=sum(x for x in s if x >= 4096),
        rows_ge512=sum(x for x in s if x >= 512)))
json.dump(out, open(sys.argv[2], 'w'), indent=0)
