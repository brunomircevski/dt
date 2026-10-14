#!/usr/bin/env bash
# Same trees: golden tests, then dumps of the old and new binaries.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64
echo "== golden tests $(date +%T)"; tests/check.sh 2>&1 | grep -v "^ok" | tail -5
W=$(mktemp -d)
for d in higgs airline; do
for alg in "--cart -d 30 --alpha 1e-5" "--c45"; do
  for run in "old/tree_cpu --parallel" "./tree_cpu --parallel" "old/tree --cuda" "./tree --cuda"; do
    $run $alg --dump $W/d bench/data/$d/train.csv > $W/o 2>&1 || echo FAILED
    printf "%-8s %-26s %-24s %s %s\n" $d "$alg" "$run" "$(sha256sum < $W/d | cut -c1-16)" "$(grep -E '^  nodes' $W/o)"
  done
done; done
echo "== done $(date +%T)"
