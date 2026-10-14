#!/usr/bin/env bash
cd "$SLURM_SUBMIT_DIR"
echo "== $(hostname) $(uname -r)"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>&1 | tr '\n' ' '; echo
cat /proc/sys/kernel/numa_balancing /sys/kernel/mm/transparent_hugepage/defrag; numactl -H | grep -E "available|distances" -A2 | head -5
sample() { for i in $(seq 1 6); do sleep 0.5; awk '{s+=$1; n++} END {printf "%.0f ", s/n/1000}' /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; done; echo "MHz avg"; }
echo "-- idle freq"; sample
echo "-- serial cart -d 12"; sample & /usr/bin/time -f "%e s %S sys" ./tree_cpu --serial --cart -d 12 --no-prune bench/data/higgs/train.csv | grep -E "build|presort|load"; wait
echo "-- serial cart -d 12 on cpu 0 with memory on node 0"; numactl -C 0 -m 0 ./tree_cpu --serial --cart -d 12 --no-prune bench/data/higgs/train.csv | grep -E "build|presort"
echo "-- parallel c45 all threads"; sample & GLIBC_TUNABLES=glibc.malloc.hugetlb=1 DT_PP=1 ./tree_cpu --parallel --threads $(nproc) --feature-parallel-rows 16384 --c45 bench/data/higgs/train.csv 2>&1 | grep -E "PP|build|presort|prune"; wait
