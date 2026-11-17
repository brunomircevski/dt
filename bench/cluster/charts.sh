#!/bin/bash
# Redraw every chart from the results in bench/results (and bench/power/results,
# paper/figs). Run it from the repository root whenever results arrive:
#
#   bench/cluster/charts.sh            # all charts
#   bench/cluster/charts.sh --no-paper # without paper/figs (needs LaTeX with cm-super)
#
# A result measured before the final protocol (bench/benchlib.py FINAL_SINCE) is
# drawn faded and labelled "out of date", under a banner naming how many; the
# marks and the banner go away by themselves once the rerun's results replace it
# and this script runs again. Copy the GPU libraries' rows first (cuML,
# XGBoost: ~/gpu-dt-bench/<results>/<dataset>/<node>/ into
# bench/results/cluster-gpu-vs-others/<dataset>/<node>/). Light work: fine on a
# login node (under nice).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-bench/.venv/bin/python}
R=bench/results
paper=1
[ "${1:-}" = --no-paper ] && paper=

for d in susy higgs mnist8m ecbdl sorel chess; do
  [ -d "$R/cluster-$d" ] && $PY bench/testset_chart.py "$R/cluster-$d"
done
# Chess against the rows: every finished run folder of these backends.
runs=()
for b in cuda-hopper cuda-ampere cpu-epyc cpu-dell serial-epyc serial-dell gpuonly-hopper gpuonly-ampere; do
  for s in 1m 10m 100m 1b; do
    [ -f "$R/cluster-chess/$b-chess_$s/report.md" ] && runs+=("$R/cluster-chess/$b-chess_$s")
  done
done
[ ${#runs[@]} -gt 0 ] && $PY bench/rows_chart.py "${runs[@]}" --out "$R/cluster-chess/chart-rows.png"
$PY bench/scaling_chart.py "$R/cluster-cpu-scaling"
$PY bench/vs_chart.py cpu
$PY bench/vs_chart.py gpu
$PY bench/vs_chart.py gpu-warm
$PY bench/e2e_chart.py
d=$R/cluster-profiling
$PY bench/profile_share_chart.py "$d" --out "$d/chart.png" > /dev/null
$PY bench/profile_share_chart.py "$d" --chess --out "$d/chart-chess.png" > /dev/null
d=$R/hybrid-vs-gpu-only
for n in ampere hopper; do
  $PY bench/hybrid_chart.py "$d" --node "$n" --datasets higgs,mnist8m,sorel_4m
  $PY bench/hybrid_profile_chart.py "$d" --node "$n" --dataset mnist8m > /dev/null
done
$PY bench/hybrid_profile_chart.py "$d" --node ampere --dataset mnist8m --hybrid-only --seconds > /dev/null
$PY bench/hybrid_profile_chart.py "$d" --node ampere --dataset mnist8m --hybrid-only > /dev/null
$PY bench/power/analyze.py bench/power/results > /dev/null
echo "bench/power/results/chart-*.png"
if [ -n "$paper" ]; then
  if kpsewhich type1ec.sty > /dev/null; then
    $PY paper/tools/charts.py
  else
    echo "paper/figs skipped: LaTeX lacks cm-super (bench/cluster/README.md: TEXMFHOME)" >&2
  fi
fi
