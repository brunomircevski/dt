#!/usr/bin/env bash
# Every cluster-<dataset> run still to do, from 2026-10-28 evening, in a few
# batch jobs (batch.sh: one SLURM job per node and group, runs one after
# another) instead of one job per run; replaces every pending job of
# submit-final- and submit-dell-chess-2026-10-28.sh (cancelled here; the
# running ones finish).
#   - GPU (CUDA hybrid, GPU only; hopper, ampere): all again on the new ./tree
#     (67495f9, the CUDA rework; installed by job 636 after its trees matched the
#     old build's), every dataset but SOREL and full ECBDL (they do not fit in
#     VRAM).
#   - CPU: dell and epyc everywhere: --simple, --serial, parallel on one
#     socket, parallel on both sockets, on the final build (tree_cpu); epyc on
#     one socket not on SOREL (256 GB per socket, it needs 340-370 GiB). Runs
#     already done on that build are kept, and --simple's (not changed by the
#     fixes).
# Runs per case: a run of about 2 minutes or less 10, under about 10 minutes
# 3, longer once. Nodes: short runs on epyc1, long ones on epyc2 (the same
# hardware); every dell run on dell1 (dell2, 512 GB, only for dell512), short
# batch first; each batch at most 12 hours. Results go to cluster-<x>-fix/
# and replace the old folders when done (bench/cluster/README.md).
# Run on the login node, from the repository root.
# As run: accuracy() first had `echo -e`, which prints nothing; the plan lines
# got ACCURACY=--test-accuracy added by hand before any batch started.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=bench/cluster/submit.sh
P=cart_depth10,c45_depth10_unpruned
C=cart_full_nodepth,c45_unpruned
BUILD=636
R=bench/results
OLD=bench/results-old-build
PLANS=bench/cluster/plans
out() { case $1 in ecbdl*) echo cluster-ecbdl-fix ;; chess*) echo cluster-chess-fix ;; *) echo cluster-$1-fix ;; esac; }
protocols() { case $1 in chess*) echo $C ;; *) echo $P ;; esac; }
accuracy() { case $1 in chess*) ;; *) printf %s -e ;; esac; }  # (echo -e prints nothing)

# Pending jobs out (the running ones finish), and their empty folders.
pending=$(squeue -a -u "$USER" -h -t PD -o %i | grep -v "^$BUILD$" || true)
[ -z "$pending" ] || scancel $pending
running=$(squeue -a -u "$USER" -h -t R -o %j)
for f in $R/cluster-*-fix/*/; do
  [ ! -e "$f/results.jsonl" ] && ! grep -qx "dt-$(basename "$f")" <<<"$running" && rm -rf "$f" || true
done
# The extra folders of the top-ups (old GPU builds, cancelled jobs) and the
# SOREL reruns: all replaced by the -fix folders.
mkdir -p $OLD
for d in $R/cluster-*-new $R/cluster-*-top $R/cluster-sorel-rerun; do
  [ -e "$d" ] || continue
  for s in "$d"/*; do  # next to the folders moved there before, never over them
    [ -e "$s" ] || continue
    to=$OLD/$(basename "$d")/$(basename "$s"); [ ! -e "$to" ] || to=$to-$(date +%H%M)
    mkdir -p "$(dirname "$to")" && mv "$s" "$to"
  done
  rmdir "$d"
done

rm -rf $PLANS && mkdir -p $PLANS
plan() {  # group runs dataset name...
  local group=$1 runs=$2 d=$3; shift 3
  $S -P $PLANS/$group.txt -m $runs $(accuracy $d) -d $d -p $(protocols $d) -o $(out $d) "$@" > /dev/null
}
simple() {  # group runs dataset node: --simple, one thread, both protocols in one folder
  local acc=ACCURACY=--test-accuracy; case $3 in chess*) acc=ACCURACY=--train-accuracy-only ;; esac
  echo "12:00:00 $(out $3) simple-$4-$3 $3 tree_simple $2 $(protocols $3) 1 43200 $acc" >> $PLANS/$1.txt
}

# GPU: everything fast (10 runs).
for g in hopper ampere; do
  for d in airline higgs susy ecbdl_15m mnist8m chess_1m chess_10m chess_100m chess_1b; do
    plan $g 10 $d cuda-$g-$d gpuonly-$g-$d
  done
done

# epyc1, short: parallel on both sockets and on one.
for d in airline higgs susy ecbdl_15m; do plan epyc1 10 $d cpu-epyc-$d socket-epyc-$d; done
plan epyc1 3 sorel cpu-epyc-sorel

# epyc2, long: --serial, then --simple's chess runs (the other --simple runs are done).
for d in airline higgs susy chess_1m chess_10m chess_100m chess_1b; do plan epyc2-serial 10 $d serial-epyc-$d; done
for d in mnist8m ecbdl_15m; do plan epyc2-serial 3 $d serial-epyc-$d; done
for d in ecbdl sorel; do plan epyc2-serial 1 $d serial-epyc-$d; done
for d in chess_1m chess_10m chess_100m; do simple epyc2-simple 10 $d epyc; done
simple epyc2-simple 1 chess_1b epyc

# dell1, short: parallel on both sockets and on one.
plan dell1 3 ecbdl cpu-dell-ecbdl socket-dell-ecbdl
plan dell1 3 sorel cpu-dell-sorel socket-dell-sorel
for d in chess_1m chess_10m chess_100m chess_1b; do plan dell1 10 $d cpu-dell-$d socket-dell-$d; done
for d in airline higgs susy ecbdl_15m mnist8m; do plan dell1 10 $d socket-dell-$d; done
# dell1, long: --serial, then --simple (full ECBDL's alone: about 4 hours).
for d in airline higgs susy chess_1m chess_10m chess_100m chess_1b; do plan dell1-serial 10 $d serial-dell-$d; done
for d in mnist8m ecbdl_15m; do plan dell1-serial 3 $d serial-dell-$d; done
for d in ecbdl sorel; do plan dell1-serial 1 $d serial-dell-$d; done
for d in susy chess_1m chess_10m chess_100m; do simple dell1-simple 10 $d dell; done
simple dell1-simple 3 higgs dell
for d in airline chess_1b ecbdl_15m; do simple dell1-simple 1 $d dell; done
simple dell1-simple-ecbdl 1 ecbdl dell

# The jobs: one per plan, the groups of a node one after another.
export TREE_GIT_DESCRIBE="67495f9-dirty"  # the binaries: 67495f9's src + the uncommitted CPU fixes
batch() {  # group partition node [dependency]
  sbatch --parsable -J dt-batch-$1 -p $2 -w $3 --exclusive -t 12:00:00 --export=ALL \
    ${4:+--dependency=$4} -o $PLANS/$1.log bench/cluster/batch.sh $PLANS/$1.txt
}
h=$(batch hopper hopper hopper afterok:$BUILD)
a=$(batch ampere ampere ampere afterok:$BUILD)
e1=$(batch epyc1 epyc epyc1)
e2=$(batch epyc2-serial epyc epyc2)
e2s=$(batch epyc2-simple epyc epyc2 afterany:$e2)
d1=$(batch dell1 dell dell1)
d1s=$(batch dell1-serial dell dell1 afterany:$d1)
d1l=$(batch dell1-simple dell dell1 afterany:$d1s)
d1e=$(batch dell1-simple-ecbdl dell dell1 afterany:$d1l)
echo "hopper $h, ampere $a, epyc1 $e1, epyc2 $e2 -> $e2s, dell1 $d1 -> $d1s -> $d1l -> $d1e"
wc -l $PLANS/*.txt
squeue -a -u "$USER" -o "%.6i %.26j %.8T %.8N %.20E"
