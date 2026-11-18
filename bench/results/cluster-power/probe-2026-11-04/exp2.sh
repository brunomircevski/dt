#!/bin/bash
# Sustained-load power experiment: exp2.sh <outdir> <cpu|cuda>
set -u
O=$1; MODE=$2; mkdir -p $O; cd ~/dt
mark() { echo "$(date +%s.%N | cut -c1-14) $*" >> $O/phases.txt; }
python3 ~/power-probe/pwrlog.py 0.1 $O/samples.csv & LOG=$!
PERF=""
if [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ]; then
  perf stat -a --per-socket -x, -I 100 -e power/energy-pkg/ -o $O/perf_rapl.csv & PERF=$!
  ls /sys/bus/event_source/devices/power_core/events/ > $O/power_core_events.txt
  perf stat -a -e power_core/energy-core/ sleep 1 > $O/power_core_test.txt 2>&1
fi
mark idle_start; sleep 20; mark idle_end
D=bench/data/higgs/train.csv
if [ $MODE = cpu ]; then
  mark "parallel_cart_m10 start"; ./tree_cpu --parallel --cart --no-prune -d 10 -m 10 $D > $O/run_par.txt 2>&1; mark "parallel_cart_m10 end"; sleep 10
  mark "parallel_c45_m10 start"; ./tree_cpu --parallel --c45 --no-prune -d 10 -m 10 $D > $O/run_c45.txt 2>&1; mark "parallel_c45_m10 end"; sleep 10
  mark "serial_cart_m2 start"; ./tree_cpu --serial --cart --no-prune -d 10 -m 2 $D > $O/run_ser.txt 2>&1; mark "serial_cart_m2 end"
else
  export CUDA_VISIBLE_DEVICES=0
  mark "cuda_cart_m20 start"; numactl --cpunodebind=0 --membind=0 ./tree --cuda --cart --no-prune -m 20 $D > $O/run_cuda.txt 2>&1; mark "cuda_cart_m20 end"; sleep 10
  mark "gpuonly_c45_m20 start"; numactl --cpunodebind=0 --membind=0 ./tree --cuda --gpu-only --c45 --no-prune -m 20 $D > $O/run_gpuonly_c45.txt 2>&1; mark "gpuonly_c45_m20 end"
fi
sleep 15; mark tail_end
kill $LOG; [ -n "$PERF" ] && kill -INT $PERF; sleep 1
