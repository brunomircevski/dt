#!/usr/bin/env bash
# Tree visualizations: chess 1M (80/20), one tree per implementation with depth
# <= 10, drawn as SVG with the run's facts, plus a timing chart. BENCHMARK.md
# explains the choices.
#
#   bench/results/visualization/run.sh            on mordor3's login node, from the repository root:
#                                                 submits every job below
#   bench/results/visualization/run.sh <role>     what one SLURM job runs
#
# Roles (each its own job, the whole node held with --exclusive):
#   prepare  dell1   the 80/20 split of datasets/chess_1m.csv into bench/data/chess_1m_8020
#   dell1    dell1   ./tree serial and parallel (48 threads), then scikit-learn and
#                    rpart side by side, pinned to one core on each socket
#   dell2    dell2   Weka J48 and YaDT (1 thread) side by side, pinned the same way,
#                    then YaDT on 48 threads
#   hopper   hopper  ./tree --cuda, cuML, XGBoost (CUDA-only libraries: the H100)
#   render   dell1   the SVGs and chart.png/svg/summary.csv, after all the others
# Results: results/<run>.json, svg/<run>.svg, chart.*, summary.csv, logs/<role>.log.
set -uo pipefail
ROLE="${1:-submit}"
if [ -n "${SLURM_JOB_ID:-}" ]; then REPO="$SLURM_SUBMIT_DIR"; else REPO="$(git rev-parse --show-toplevel)"; fi
V="$REPO/bench/results/visualization"
DATA="$REPO/bench/data/chess_1m_8020"
RES="$V/results"
CPU_PY="$REPO/bench/.venv/bin/python"
GPU_PY="${GPU_PY:-$HOME/gpu-dt-bench/venv/bin/python}"
REPS="${REPS:-3}"
DEPTH=10
FIT="$V/scripts/fit.py"
export PATH="${BENCH_ENV:-$HOME/micromamba/envs/dtbench}/bin:$PATH" PYTHONUNBUFFERED=1
step() { echo; echo "=== $(date +%T) $*"; }
fit() { timeout --foreground 1800 "$@" || echo "FAILED ($?): $*"; }

case "$ROLE" in
submit)
  cd "$REPO"
  mkdir -p "$V/logs" "$RES"
  desc="$(git describe --always --dirty --abbrev=12)"
  sub() { sbatch --parsable -J "viz-$1" -p "$2" -w "$3" --exclusive -t "$4" -o "$V/logs/$1.log" \
            --export=ALL,TREE_GIT_DESCRIBE="$desc" ${5:+-d "$5"} "$V/run.sh" "$1"; }
  p=$(sub prepare dell dell1 00:15:00)
  a=$(sub dell1 dell dell1 01:00:00 "afterok:$p")
  b=$(sub dell2 dell dell2 01:00:00 "afterok:$p")
  g=$(sub hopper hopper hopper 01:00:00 "afterok:$p")
  r=$(sub render dell dell1 00:15:00 "afterany:$a:$b:$g")
  echo "jobs: prepare $p, dell1 $a, dell2 $b, hopper $g, render $r (code $desc)"
  ;;
prepare)
  step "split $REPO/datasets/chess_1m.csv"
  src="$REPO/datasets/chess_1m.csv"
  [ -e "$src" ] || src="$HOME/dt/datasets/chess_1m.csv"
  "$CPU_PY" "$V/scripts/prepare.py" "$src" "$DATA"
  ;;
dell1)
  step "build ./tree_cpu on $(hostname)"
  make -C "$REPO" -j "$(nproc)" tree_cpu >/dev/null || exit 1
  export TREE_BIN="$REPO/tree_cpu"
  for c in cart c45; do
    step "./tree serial $c"; fit "$CPU_PY" "$FIT" tree $c "$DATA" "$RES" --backend serial --reps "$REPS" --depth $DEPTH
    step "./tree parallel $c"; fit "$CPU_PY" "$FIT" tree $c "$DATA" "$RES" --backend parallel --threads "$(nproc)" --reps "$REPS" --depth $DEPTH
  done
  step "scikit-learn (CPU 0) and rpart (CPU $(($(nproc) / 2))) side by side"
  fit taskset -c 0 "$CPU_PY" "$FIT" sklearn cart "$DATA" "$RES" --reps "$REPS" --depth $DEPTH &
  fit taskset -c $(($(nproc) / 2)) "$CPU_PY" "$FIT" rpart cart "$DATA" "$RES" --reps "$REPS" --depth $DEPTH &
  wait
  ;;
dell2)
  classes="$(mktemp -d)"
  step "compile J48Export"
  javac -cp "$REPO/bench/.tools/weka.jar" -d "$classes" "$V/scripts/J48Export.java" || exit 1
  step "Weka J48 (CPU 0) and YaDT 1 thread (CPU $(($(nproc) / 2))) side by side"
  fit taskset -c 0 "$CPU_PY" "$FIT" j48 c45 "$DATA" "$RES" --reps "$REPS" --depth $DEPTH --classes "$classes" &
  fit taskset -c $(($(nproc) / 2)) "$CPU_PY" "$FIT" yadt c45 "$DATA" "$RES" --threads 1 --reps "$REPS" --depth $DEPTH &
  wait
  step "YaDT $(nproc) threads"
  fit "$CPU_PY" "$FIT" yadt c45 "$DATA" "$RES" --threads "$(nproc)" --reps "$REPS" --depth $DEPTH
  rm -r "$classes"
  ;;
hopper)
  export CUDA_VISIBLE_DEVICES=0
  step "build ./tree (CUDA, sm_90) on $(hostname)"
  make -C "$REPO" -j "$(nproc)" tree NVCC=/usr/local/cuda/bin/nvcc CUDA_LIB=/usr/local/cuda/lib64 \
    CUDA_ARCH="-gencode arch=compute_90,code=sm_90" >/dev/null || exit 1
  export TREE_BIN="$REPO/tree"
  for c in cart c45; do
    step "./tree --cuda $c"; fit "$CPU_PY" "$FIT" tree $c "$DATA" "$RES" --backend cuda --threads "$(nproc)" --reps "$REPS" --depth $DEPTH
    step "cuML $c"; fit "$GPU_PY" "$FIT" cuml $c "$DATA" "$RES" --reps "$REPS" --depth $DEPTH
  done
  step "XGBoost cart"; fit "$GPU_PY" "$FIT" xgboost cart "$DATA" "$RES" --reps "$REPS" --depth $DEPTH
  ;;
render)
  step "SVGs"
  "$CPU_PY" "$V/scripts/render.py" "$RES"/*.json --out "$V/svg"
  step "chart"
  "$CPU_PY" "$V/scripts/chart.py" "$RES" "$V"
  ;;
*) sed -n '2,20p' "$0"; exit 2 ;;
esac
echo "done $(date '+%F %T')"
