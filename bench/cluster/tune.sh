#!/usr/bin/env bash
# Tuning runs on one whole node (SLURM job): every case of a case file, `reps`
# times, the cases interleaved (case 1, 2, ..., then again), so a slow phase of
# the node does not hit one case only. Prints one CSV line per run with
# ./tree's own phase timings, /usr/bin/time's CPU times, minor page faults and
# context switches (bench/results/cluster-tuning/ has the sweeps made with it):
#
#   sbatch -p epyc --exclusive -o out.csv bench/cluster/tune.sh <case file> <reps>
#
# A case file has one case per line, "label | environment | arguments", e.g.
#   cart-t48 | | ./tree_cpu --parallel --threads 48 --cart -d 30 --alpha 1e-5 bench/data/higgs/train.csv
# Lines starting with # are skipped. Arguments may use $CORES (the node's CPUs).
# TUNE_KEEP=1 also copies ./tree's output to stderr (e.g. for --gpu-profile).

set -euo pipefail
CASES="$1" REPS="${2:-3}"
cd "$SLURM_SUBMIT_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CORES="$(nproc)"
echo "# $(hostname), $CORES CPUs, job ${SLURM_JOB_ID:-}, $(date '+%F %T'), load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "label,rep,load_ms,setup_ms,presort_ms,build_ms,prune_ms,train_ms,nodes,depth,real_s,user_s,sys_s,minor_faults,voluntary_cs,involuntary_cs"
mapfile -t lines < <(grep -v '^\s*#' "$CASES" | grep -v '^\s*$')
for ((rep = 0; rep < REPS; rep++)); do
  for line in "${lines[@]}"; do
    IFS='|' read -r label env args <<<"$line"
    label="$(echo $label)"
    out="$(mktemp)"
    # shellcheck disable=SC2086
    if ! /usr/bin/env $env /usr/bin/time -o "$out.time" -f "%e %U %S %R %w %c" \
        bash -c "exec ${args//\$CORES/$CORES}" >"$out.log" 2>&1; then
      echo "$label,$rep,FAILED: $(tail -1 "$out.log")"
      rm -f "$out" "$out.log" "$out.time"
      continue
    fi
    awk -v label="$label" -v rep="$rep" -v times="$(tail -1 "$out.time")" '
      /^  load /        { load = $2 }
      /^  gpu setup /   { setup = $3 }
      /^  presort /     { presort = $2 }
      /^  build /       { build = $2 }
      /^  prune /       { prune = $2 }
      /^  train total / { train = $3 }
      /^  nodes /       { nodes = $2; sub(",", "", nodes); depth = $6 }
      END { split(times, t, " ");
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", label, rep, load, setup,
                   presort, build, prune, train, nodes, depth, t[1], t[2], t[3], t[4], t[5], t[6] }' "$out.log"
    if [ -n "${TUNE_KEEP:-}" ]; then
      sed "s/^/  [$label] /" "$out.log" >&2
    fi
    rm -f "$out" "$out.log" "$out.time"
  done
done
echo "# done $(date '+%F %T')"
