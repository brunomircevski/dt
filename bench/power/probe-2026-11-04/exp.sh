#!/bin/bash
# Short power experiment: idle, workload phases, idle. Phase marks go to phases.txt.
#   sbatch ... exp.sh <outdir> <mode: cpu|cuda>
set -u
O=$1; MODE=$2; mkdir -p $O; cd ~/dt
mark() { echo "$(date +%s.%N | cut -c1-14) $*" >> $O/phases.txt; }
for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = power_meter ] && for f in $h/power1_*; do echo "$f: $(cat $f 2>&1)"; done; done > $O/power_meter_attrs.txt
python3 ~/power-probe/pwrlog.py 0.1 $O/samples.csv & LOG=$!
PERF=""
if [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ]; then
  perf stat -a -x, -I 100 -e power/energy-pkg/ -o $O/perf_rapl.csv & PERF=$!
  perf stat -a --per-socket -e power/energy-pkg/ -o $O/perf_rapl_total_socket.txt sleep 0 2>/dev/null
fi
mark idle_start; sleep 15; mark idle_end
D=bench/data/higgs/train.csv
if [ $MODE = cpu ]; then
  for i in 1 2; do mark "parallel_cart_$i start"; ./tree_cpu --parallel --cart --no-prune -d 10 $D > $O/run_par_$i.txt 2>&1; mark "parallel_cart_$i end"; sleep 5; done
  mark "parallel_c45 start"; ./tree_cpu --parallel --c45 --no-prune -d 10 $D > $O/run_c45.txt 2>&1; mark "parallel_c45 end"; sleep 5
  mark "serial_cart start"; ./tree_cpu --serial --cart --no-prune -d 10 $D > $O/run_ser.txt 2>&1; mark "serial_cart end"
else
  export CUDA_VISIBLE_DEVICES=0
  for i in 1 2 3; do mark "cuda_cart_$i start"; ./tree --cuda --cart --no-prune -d 10 $D > $O/run_cuda_$i.txt 2>&1; mark "cuda_cart_$i end"; sleep 5; done
  mark "cuda_c45 start"; ./tree --cuda --c45 --no-prune -d 10 $D > $O/run_cuda_c45.txt 2>&1; mark "cuda_c45 end"
fi
sleep 10; mark tail_end
kill $LOG; [ -n "$PERF" ] && kill -INT $PERF; sleep 1
