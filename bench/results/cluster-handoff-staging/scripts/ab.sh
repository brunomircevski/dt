#!/usr/bin/env bash
# Final A/B: base (e0a9106) vs new5 (staged hand-offs, 128 MB drains, ordered touch): base/tree (arena page-locked, e0a9106) vs
# new/tree (staged through page-locked buffers). Submit from ~/dt-staged.
#SBATCH -t 03:00:00
#SBATCH --exclusive
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
node=$(hostname -s); T=$(nproc); OUT=ab5/$node; mkdir -p $OUT
echo "job $SLURM_JOB_ID on $node, $T threads, $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -1
numactl -H 2>/dev/null | grep -E "free|size" | head -4
run() { # label dataset flags...
  local label=$1 data=$2; shift 2
  local tag=$(echo "$*" | tr -d ' -.')
  local f=$OUT/$data-$tag-$label-$(date +%H%M%S)
  taskset -c 0-$((T-1)) ${BIN:-$label}/tree --cuda --threads $T "$@" --profile $f.json \
    --dump $f.dump base/bench/data/$data/train.csv > $f.out 2>&1
  local sum=$(md5sum < $f.dump | cut -c1-12); rm -f $f.dump
  python3 - "$f.json" "$label" "$data $*" "$sum" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); g=d['tables']['gpu_levels']
ph={p['name']:p['end']-p['start'] for p in d['phases'] if p['depth']==0}
print('%-6s %-30s train %7.2f setup %.2f grow-setup %.2f L0 %.2f levels %6.2f wait %.2f arrange %.2f nodes %d tree %s' % (
  sys.argv[2], sys.argv[3], d['train_seconds'], ph.get('gpu setup',0), ph.get('grow setup',0),
  g[0]['seconds'] if g else 0, ph.get('gpu levels',0), ph.get('wait cpu subtrees',0),
  ph.get('prune arrange rows',0), d['nodes'], sys.argv[4]), flush=True)
PY
}
if command -v /usr/local/cuda/bin/compute-sanitizer >/dev/null; then
  for p in "--c45" "--cart --no-prune"; do
    /usr/local/cuda/bin/compute-sanitizer --tool memcheck --print-limit 5 new5/tree --cuda --threads $T $p \
      base/bench/data/susy_500k/train.csv 2>&1 | grep -E "ERROR SUMMARY|train total" | sed "s/^/sanitizer $p: /"
  done
fi
for rep in 1 2 3; do
  for p in "--c45" "--cart --no-prune" "--cart --alpha 1e-05"; do
    for d in airline higgs; do
      run base $d $p
      run new5 $d $p
    done
  done
done
for rep in 1 2; do
  for p in "--c45" "--cart --no-prune" "--cart --alpha 1e-05"; do
    run base ecbdl_15m $p
    run new5 ecbdl_15m $p
  done
done
echo "done $(date '+%F %T')"
