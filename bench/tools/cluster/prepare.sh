#!/usr/bin/env bash
# SLURM job on a CPU node of mordor3: rebuild the datasets from the public
# downloads (checked by SHA-256) and prepare one of them for bench/tools/run.py
# (PREPARE, a name in bench/tools/datasets.toml; default higgs). Submit it from the
# repository root (never run it on the login node); arguments choose the
# datasets to rebuild (default: all). PREPARE may name several, e.g. ECBDL
# (needs about 350 GB of RAM: run it on a 1 TB node, ampere or hopper):
#
#   sbatch bench/tools/cluster/prepare.sh [covertype higgs supersymmetry]
#   sbatch --export=ALL,PREPARE=susy -t 02:00:00 \
#       -o bench/tools/cluster/prepare-susy.log bench/tools/cluster/prepare.sh supersymmetry
#   sbatch -p ampere -w ampere --exclusive -c 48 -t 08:00:00 \
#       --export=ALL,PREPARE="ecbdl ecbdl_15m" -o bench/tools/cluster/prepare-ecbdl.log \
#       bench/tools/cluster/prepare.sh ecbdl
#
#SBATCH -J dt-prepare
#SBATCH -p dell
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o bench/tools/cluster/prepare-higgs.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=bench/tools/.venv/bin/python
PREPARE="${PREPARE:-higgs}"
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, $(date '+%F %T')"

step "1/3 source CSVs from the downloads in each dataset's folder, checked by SHA-256"
"$PY" -I bench/tools/cluster/make_datasets.py "$@"

step "2/3 prepare $PREPARE for bench/tools/run.py"
# shellcheck disable=SC2086  # PREPARE: one or more names
"$PY" bench/tools/prepare.py $PREPARE

step "3/3 SHA-256 of the prepared files"
for name in $PREPARE; do  # a set of another's first rows is only <parent>/<name>.meta.json
  if [ -d "bench/tools/data/$name" ]; then (cd "bench/tools/data/$name" && sha256sum meta.json train.* test.* yadt.names 2>/dev/null || true)
  else (cd bench/tools/data && sha256sum */"$name".meta.json 2>/dev/null || true); fi
done
echo "done $(date '+%F %T')"
