#!/usr/bin/env bash
# Several benchmarks one after another in one SLURM job on one whole node: the
# lines of a plan file (submit.sh -P), each run as its own job.sh would run it,
# its log in its folder's run.log. Fewer, longer jobs: one per node and group
# (short runs, long runs), instead of one per run. A run that fails does not
# stop the next. Submit it from the repository root, e.g.
#
#   sbatch -p dell -w dell3 --exclusive -t 12:00:00 -o bench/cluster/plans/dell3.log \
#       bench/cluster/batch.sh bench/cluster/plans/dell3.txt
#
# Plan line: <time limit> <dir> <name> <dataset> <impls> <runs> <protocols> <threads>
#            <timeout> [VAR=value,VAR=value]  (the limit is the submitter's)
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# A node that was idle runs its first seconds of work slow (dell3, powersave
# governor, 2026-11-15: the first 1-3 processes of a job 10-30% slower, with
# as much more CPU time): keep every CPU busy for 30 s before the first run.
for _ in $(seq "$(nproc)"); do timeout 30 sh -c 'while :; do :; done' & done
wait
failed=()
while read -r _ dir name dataset impls runs protocols threads timeout environment <&3; do
  [ -n "$dir" ] || continue
  echo "$(date '+%F %T')  $dir/$name"
  mkdir -p "bench/results/$dir/$name"
  # shellcheck disable=SC2086
  env ${environment//,/ } bench/cluster/job.sh "$dir" "$name" "$dataset" "$impls" "$runs" "$protocols" \
    "$threads" "$timeout" > "bench/results/$dir/$name/run.log" 2>&1 < /dev/null ||
    { echo "  failed (exit $?): bench/results/$dir/$name/run.log"; failed+=("$name"); }
done 3< "$1"
echo "$(date '+%F %T')  done${failed[*]:+, failed: ${failed[*]}}"
[ ${#failed[@]} -eq 0 ]
