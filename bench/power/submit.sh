#!/usr/bin/env bash
# Submit power_bench.sh jobs, one per node, from the repository root (on the
# mordor3 login node):
#
#   bench/power/submit.sh [-o OUT] [-d JOBID] [-c] NODE... [VAR=value...]
#
#   -o OUT    results folder (default bench/power/results); the job writes OUT/<node>/
#   -d JOBID  start after that job ends (afterany), e.g. the node's last time-benchmark job
#   -c        the warm-up check alone (CHECK_ONLY=1, GPU nodes), into OUT/check-<node>/
#   VAR=value power_bench.sh settings for every job given (they override the node's)
#
# NODE: epyc1, dell4 (kind cpu), ampere, hopper (kind all). The environment goes
# into the job's command line (sbatch --wrap), not through sbatch --export,
# which can leave a job held on mordor3. ./tree and ./tree_cpu must be built
# (bench/cluster/build.sh) at the commit in TREE_GIT_DESCRIBE (default: git
# describe of this checkout).
set -euo pipefail
OUT=bench/power/results DEP= CHECK=
while getopts "o:d:c" option; do
  case "$option" in
    o) OUT="$OPTARG" ;;
    d) DEP="$OPTARG" ;;
    c) CHECK=1 ;;
    *) sed -n '2,18p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
NODES=() EXTRA=()
for a in "$@"; do
  case "$a" in *=*) EXTRA+=("$a") ;; *) NODES+=("$a") ;; esac
done
[ ${#NODES[@]} -gt 0 ] || { sed -n '2,18p' "$0"; exit 2; }
DESCRIBE="${TREE_GIT_DESCRIBE:-$(git describe --always --dirty --abbrev=12)}"
env=()
mkdir -p "$OUT"
for node in "${NODES[@]}"; do
  # partition, time limit, kind, the node's settings
  case "$node" in
    epyc1|epyc2) p=epyc t=8:00:00 kind=cpu env=() ;;
    dell[1-4]) p=dell t=10:00:00 kind=cpu env=() ;;
    # ampere's meter takes a point reading once a minute: longer processes,
    # longer staircase levels, so that the meter samples every phase
    ampere) p=ampere t=16:00:00 kind=all env=(TARGET=240 CAL=240 IDLE=600) ;;
    hopper) p=hopper t=12:00:00 kind=all env=() ;;
    *) echo "unknown node $node" >&2; exit 2 ;;
  esac
  dir="$OUT/$node" name="pwr-$node"
  if [ -n "$CHECK" ]; then
    [ "$kind" = all ] || { echo "-c: $node has no GPU" >&2; exit 2; }
    dir="$OUT/check-$node" name="pwr-check-$node" env+=(CHECK_ONLY=1) t=1:30:00
  fi
  [ ! -e "$dir/markers.txt" ] || { echo "$dir has a job's results already" >&2; exit 1; }
  mkdir -p "$dir"
  cmd="env $(printf '%q ' "TREE_GIT_DESCRIBE=$DESCRIBE" "${env[@]}" "${EXTRA[@]}")bench/power/power_bench.sh $kind $(printf %q "$dir")"
  id=$(sbatch --parsable -p "$p" -w "$node" --exclusive -t "$t" -J "$name" ${DEP:+-d afterany:$DEP} \
         -o "$dir/job.log" --wrap "$cmd")
  echo "$node: job $id ($cmd)"
done
