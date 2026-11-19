#!/bin/bash
# Redraw every chart from the results in bench/ (and bench/cluster-power,
# paper/figs). Run it from the repository root whenever results arrive:
#
#   bench/tools/cluster/charts.sh            # all charts
#   bench/tools/cluster/charts.sh --no-paper # without paper/figs (needs LaTeX with cm-super)
#
# A result measured before the final protocol (bench/tools/benchlib.py FINAL_SINCE) is
# drawn faded and labelled "out of date", under a banner naming how many; the
# marks and the banner go away by themselves once the rerun's results replace it
# and this script runs again. Light work: fine on a login node (under nice).
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-bench/tools/.venv/bin/python}
R=bench
paper=1
[ "${1:-}" = --no-paper ] && paper=

# ./tree's epyc runs of the vs datasets, measured with the libraries: copied into
# cluster-<data> (final.sh does not run them twice)
for v in "$R"/cluster-cpu-vs-others/vs-epyc-*/; do
  [ -e "$v/results.jsonl" ] && $PY bench/tools/cluster/have_results.py split "$v"
done
for d in susy higgs mnist8m ecbdl sorel chess; do
  [ -d "$R/cluster-$d" ] && $PY bench/tools/testset_chart.py "$R/cluster-$d"
done
# Chess against the rows: every finished run folder of these backends.
runs=()
for b in cuda-hopper cuda-ampere cpu-epyc cpu-dell serial-epyc serial-dell gpuonly-hopper gpuonly-ampere; do
  for s in 1m 10m 100m 1b; do
    [ -f "$R/cluster-chess/$b-chess_$s/report.md" ] && runs+=("$R/cluster-chess/$b-chess_$s")
  done
done
[ ${#runs[@]} -gt 0 ] && $PY bench/tools/rows_chart.py "${runs[@]}" --out "$R/cluster-chess/chart-rows.png"
$PY bench/tools/scaling_chart.py "$R/cluster-cpu-scaling"
$PY bench/tools/vs_chart.py cpu
$PY bench/tools/gpu_steps_chart.py
d=$R/cluster-profiling
$PY bench/tools/profile_share_chart.py "$d" --out "$d/chart.png" > /dev/null
$PY bench/tools/profile_share_chart.py "$d" --chess --out "$d/chart-chess.png" > /dev/null
d=$R/hybrid-vs-gpu-only
for n in ampere hopper; do
  $PY bench/tools/hybrid_chart.py "$d" --node "$n" --datasets higgs,mnist8m,sorel_4m
  $PY bench/tools/hybrid_profile_chart.py "$d" --node "$n" --dataset mnist8m > /dev/null
done
$PY bench/tools/hybrid_profile_chart.py "$d" --node ampere --dataset mnist8m --hybrid-only --seconds > /dev/null
$PY bench/tools/hybrid_profile_chart.py "$d" --node ampere --dataset mnist8m --hybrid-only > /dev/null
$PY bench/tools/power/analyze.py bench/cluster-power > /dev/null
echo "bench/cluster-power/chart-*.png"
if [ -n "$paper" ]; then
  if kpsewhich type1ec.sty > /dev/null; then
    $PY paper/tools/charts.py
  else
    echo "paper/figs skipped: LaTeX lacks cm-super (bench/tools/cluster/README.md: TEXMFHOME)" >&2
  fi
fi
