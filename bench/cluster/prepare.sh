#!/usr/bin/env bash
# SLURM job on a CPU node of mordor3: rebuild the datasets from the public
# downloads (checked by SHA-256) and prepare one of them for bench/run.py
# (PREPARE, a name in bench/datasets.toml; default higgs). Submit it from the
# repository root (never run it on the login node); arguments choose the
# datasets to rebuild (default: all):
#
#   sbatch bench/cluster/prepare.sh [covertype higgs supersymmetry airline]
#   sbatch --export=ALL,PREPARE=airline -t 02:00:00 \
#       -o bench/cluster/prepare-airline.log bench/cluster/prepare.sh airline
#
#SBATCH -J dt-prepare
#SBATCH -p dell
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o bench/cluster/prepare-higgs.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=bench/.venv/bin/python
DOWNLOADS="${DOWNLOADS:-$HOME/downloads}"
PREPARE="${PREPARE:-higgs}"
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, $(date '+%F %T')"

step "1/3 datasets from $DOWNLOADS, checked by SHA-256"
"$PY" -I bench/cluster/make_datasets.py "$DOWNLOADS" "$@"

step "2/3 prepare $PREPARE for bench/run.py"
"$PY" bench/prepare.py "$PREPARE"

step "3/3 SHA-256 of the prepared files (for higgs: compare with plan.json of pc-legion-higgs/cuda-legion-higgs)"
sha256sum "bench/data/$PREPARE"/*
echo "done $(date '+%F %T')"
