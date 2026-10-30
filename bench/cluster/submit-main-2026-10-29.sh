#!/usr/bin/env bash
# Every ./tree run that main's CPU presort speed-up (4fb419c) made out of date,
# again on one build of main (ec77727, 2026-10-29 night): ~/dt/tree_main (CPU +
# CUDA) and ~/dt/tree_cpu_main, built from `git archive ec77727` in ~/dt-main by
# build.sh (job 880). Every tree at depth <= 10 without pruning (cart_depth10,
# c45_depth10_unpruned), chess included (it had cart_full_nodepth/c45_unpruned:
# the same depth-4 trees). Replaces submit-cpu-2026-10-29.sh's jobs 797/799
# (cancelled: they ran the old binary); job 798's chess 1000M --simple and the
# --simple runs of jobs 797/798 are kept (--simple's code did not change).
#   - Test-set benchmarks (cluster-<x>): parallel on both sockets and on one,
#     --serial, CUDA hybrid and GPU only; test accuracy and tree hashes in every
#     run. --simple only where missing (chess at depth 10).
#   - cluster-cpu-scaling: HIGGS and MNIST8M, 1, 2, 4, then every 4 threads to
#     44 (48: the test-set benchmark's run).
#   - cluster-profiling: epyc, dell1 and hopper (hybrid and GPU only), profiled;
#     plain runs only for chess 50M, 500M and 2000M (the others are the
#     test-set benchmarks' runs).
#   - cluster-cpu-vs-others: Covertype's ./tree bars (chess's come from
#     cluster-chess), and scikit-learn and J48 again with 3 runs where a run
#     takes under 10 minutes.
# Runs per case: 10 when a run takes about 2 minutes or less (chess 30), 3
# under about 10 minutes, longer ones once; SOREL's parallel runs 3 (each under
# 5 minutes of training). Order on each node: short cases first. Nodes: epyc1
# and epyc2 (the same hardware) share the epyc work; dell1 (1 TB) every dell
# run, dell2 the 512 GB chess runs; hopper and ampere the CUDA runs. Into
# cluster-<x>-main/, moved into place when done. Run on the login node, from
# the repository root. Run as jobs 881-886 (epyc1, epyc2, dell1, dell2, hopper,
# ampere); top-ups: bench/cluster/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=cart_depth10,c45_depth10_unpruned
PLANS=bench/cluster/plans
BIN="TREE_BIN=$PWD/tree_cpu_main,TREE_CUDA_BIN=$PWD/tree_main,TREE_GIT_DESCRIBE=ec77727"
TS=",ACCURACY=--test-accuracy,RUN_EXTRA=--hash-trees"
THREADS="4,$(seq -s, 8 4 44)"
for f in tree_main tree_cpu_main; do [ -x $f ] || { echo "$f missing" >&2; exit 1; }; done
GROUPS_=(epyc1-main epyc2-main dell1-main dell2-main hopper-main ampere-main)
for g in "${GROUPS_[@]}"; do rm -f $PLANS/$g.txt; done

line() {  # plan dir name dataset impls runs threads env [protocols]
  echo "_ $2 $3 $4 $5 $6 ${9:-$P} $7 18000 $8" >> $PLANS/$1.txt
}
out() { case $1 in ecbdl*) echo cluster-ecbdl-main ;; chess*) echo cluster-chess-main ;; *) echo cluster-$1-main ;; esac; }
runs() {  # backend dataset -> runs per case
  case $1:$2 in
    simple:chess_1m|simple:chess_10m) echo 30 ;;
    simple:chess_100m) echo 3 ;;
    simple:*) echo 1 ;;
    serial:chess_1b|serial:mnist8m|serial:ecbdl_15m) echo 3 ;;
    serial:ecbdl|serial:sorel) echo 1 ;;
    *:chess*) echo 30 ;;
    *:ecbdl|*:sorel) echo 3 ;;
    *) echo 10 ;;
  esac
}
ts() {  # plan backend node datasets...: the test-set benchmarks
  local plan=$1 b=$2 node=$3 d; shift 3
  for d in "$@"; do
    case $b in
      cpu) line $plan $(out $d) cpu-$node-$d $d tree $(runs $b $d) all "$BIN$TS" ;;
      socket) line $plan $(out $d) socket-$node-$d $d tree $(runs $b $d) all "$BIN$TS,SOCKET=auto" ;;
      serial) line $plan $(out $d) serial-$node-$d $d tree $(runs $b $d) 1 "$BIN$TS" ;;
      simple) line $plan $(out $d) simple-$node-$d $d tree_simple $(runs $b $d) 1 "$BIN$TS" ;;
      cuda) line $plan $(out $d) cuda-$node-$d $d tree_cuda $(runs $b $d) all "$BIN$TS" ;;
      gpuonly) line $plan $(out $d) gpuonly-$node-$d $d tree_gpu_only $(runs $b $d) all "$BIN$TS" ;;
    esac
  done
}
profile() {  # plan mode node runs plain(yes/no) datasets...
  local plan=$1 mode=$2 node=$3 n=$4 plain=$5 impl d; shift 5
  case $mode in cpu) impl=tree ;; cuda) impl=tree_cuda ;; gpuonly) impl=tree_gpu_only ;; esac
  for d in "$@"; do
    line $plan cluster-profiling-main $mode-$node-$d $d $impl $n all "$BIN,RUN_EXTRA=--profile"
    [ $plain = no ] || line $plan cluster-profiling-main/plain $mode-$node-$d $d $impl $n all "$BIN$TS"
  done
}
scaling() {  # plan node dataset runs threads [dir]
  line $1 ${6:-cluster-cpu-scaling-main} cpu-$2-$3 $3 tree $4 $5 "$BIN"
}
library() {  # plan dataset tool protocol: scikit-learn or J48 on one socket, 3 runs
  local impl=$3 timeout=21600 p=cart_depth10
  [ $4 = c45 ] && p=c45_depth10_unpruned
  echo "_ cluster-cpu-vs-others-main vs-epyc-$2-$3-$4 $2 $impl 3 $p 1 86400 $BIN$TS,SOCKET=auto" >> $PLANS/$1.txt
}
CHESS="chess_1m chess_10m chess_100m"
SHORT="susy higgs airline"

# epyc1: the short cases, then HIGGS scaling, chess 1000M, profiling, ECBDL 15M.
for b in cpu socket; do ts epyc1-main $b epyc $CHESS $SHORT; done
ts epyc1-main serial epyc $CHESS susy higgs
ts epyc1-main serial epyc airline chess_1b mnist8m
ts epyc1-main simple epyc chess_1m chess_100m
line epyc1-main cluster-cpu-vs-others-main cpu-epyc-covertype covertype tree 10 all "$BIN$TS"
line epyc1-main cluster-cpu-vs-others-main socket-epyc-covertype covertype tree 10 all "$BIN$TS,SOCKET=auto"
line epyc1-main cluster-cpu-vs-others-main serial-epyc-covertype covertype tree 10 1 "$BIN$TS"
for d in covertype chess_10m; do library epyc1-main $d sklearn cart; library epyc1-main $d j48 c45; done
profile epyc1-main cpu epyc 10 no higgs airline
profile epyc1-main cpu epyc 30 no chess_10m chess_100m
profile epyc1-main cpu epyc 30 yes chess_50m chess_500m
scaling epyc1-main epyc higgs 10 "1,2,$THREADS"
for b in cpu socket; do ts epyc1-main $b epyc chess_1b; done
profile epyc1-main cpu epyc 30 no chess_1b
profile epyc1-main cpu epyc 10 no ecbdl_15m
for b in cpu socket; do ts epyc1-main $b epyc ecbdl_15m mnist8m; done
profile epyc1-main cpu epyc 30 yes chess_2b

# epyc2 (after job 798): libraries on SUSY and HIGGS, ECBDL 15M --serial,
# MNIST8M scaling, full ECBDL and SOREL.
library epyc2-main susy sklearn cart; library epyc2-main susy j48 c45; library epyc2-main higgs sklearn cart
ts epyc2-main serial epyc ecbdl_15m
scaling epyc2-main epyc mnist8m 10 "$THREADS"
scaling epyc2-main epyc mnist8m 3 1,2 cluster-cpu-scaling-main2
for b in cpu socket; do ts epyc2-main $b epyc ecbdl; done
ts epyc2-main serial epyc ecbdl
ts epyc2-main cpu epyc sorel
ts epyc2-main serial epyc sorel

# dell1: everything dell, short cases first.
for b in cpu socket; do ts dell1-main $b dell $CHESS $SHORT; done
ts dell1-main serial dell $CHESS susy higgs
ts dell1-main simple dell chess_1m chess_10m chess_100m
profile dell1-main cpu dell 10 no higgs airline
profile dell1-main cpu dell 30 no chess_10m chess_100m
profile dell1-main cpu dell 30 yes chess_50m chess_500m
scaling dell1-main dell higgs 10 "1,2,$THREADS"
for b in cpu socket; do ts dell1-main $b dell chess_1b ecbdl_15m mnist8m; done
profile dell1-main cpu dell 30 no chess_1b
profile dell1-main cpu dell 10 no ecbdl_15m
ts dell1-main serial dell airline chess_1b mnist8m ecbdl_15m
for b in cpu socket; do ts dell1-main $b dell ecbdl sorel; done
scaling dell1-main dell mnist8m 10 "$THREADS"
scaling dell1-main dell mnist8m 3 1,2 cluster-cpu-scaling-main2
profile dell1-main cpu dell 30 yes chess_2b
ts dell1-main simple dell chess_1b
ts dell1-main serial dell ecbdl sorel

# dell2: the 512 GB node's chess runs.
ts dell2-main cpu dell512 $CHESS chess_1b

# hopper: the test-set runs, short first, then profiling; ampere: the test-set runs.
for g in hopper ampere; do
  for d in $CHESS susy higgs airline mnist8m ecbdl_15m chess_1b; do ts $g-main cuda $g $d; ts $g-main gpuonly $g $d; done
done
for m in cuda gpuonly; do
  profile hopper-main $m hopper 10 no higgs airline ecbdl_15m
  profile hopper-main $m hopper 30 no chess_10m chess_100m chess_1b
  profile hopper-main $m hopper 30 yes chess_50m chess_500m chess_2b
done

batch() {  # group partition node [dependency]
  local limit=7-00:00:00; [ $2 != dell-512GB ] || limit=1-00:00:00
  sbatch --parsable -J dt-batch-$1 -p $2 -w $3 --exclusive -t $limit --export=ALL \
    ${4:+--dependency=$4} -o $PLANS/$1.log bench/cluster/batch.sh $PLANS/$1.txt
}
before=$(squeue -a -h -u "$USER" -n dt-batch-epyc2-cpu -o %i)
echo "epyc1 $(batch epyc1-main epyc epyc1), epyc2 $(batch epyc2-main epyc epyc2 ${before:+afterany:$before})" \
  "(after ${before:-nothing}), dell1 $(batch dell1-main dell-long dell1), dell2 $(batch dell2-main dell-512GB dell2)," \
  "hopper $(batch hopper-main hopper hopper), ampere $(batch ampere-main ampere ampere)"
wc -l $PLANS/*-main.txt
squeue -a -u "$USER" -o "%.6i %.26j %.8T %.8N %.20E"
