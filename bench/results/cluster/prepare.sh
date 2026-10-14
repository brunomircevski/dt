#!/usr/bin/env bash
# SLURM job on a CPU node of mordor3: rebuild the datasets from the public
# downloads (checked against the desktop's SHA-256) and prepare HIGGS for
# bench/run.py. Submit it from the repository root (never run it on the login
# node); arguments choose the datasets (default: all):
#
#   sbatch bench/results/cluster/prepare.sh [covertype higgs supersymmetry]
#
#SBATCH -J dt-prepare
#SBATCH -p dell
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o bench/results/cluster/prepare.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=bench/.venv/bin/python
DOWNLOADS="${DOWNLOADS:-$HOME/downloads}"
step() { echo; echo "=== $(date +%H:%M:%S)  $*"; }
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, $(date '+%F %T')"

step "1/3 datasets from $DOWNLOADS, checked against the desktop's SHA-256"
"$PY" -I bench/results/cluster/make_datasets.py "$DOWNLOADS" "$@"

step "2/3 prepare HIGGS for bench/run.py"
"$PY" bench/prepare.py higgs

step "3/3 SHA-256 of the prepared files (compare with plan.json of cuda-legion-higgs)"
sha256sum bench/data/higgs/*
echo "done $(date '+%F %T')"
