#!/bin/bash
#SBATCH -p dell -w dell2 --exclusive -t 1:00:00 -J dellclk
#SBATCH -o bench/cluster-cpu-scaling/dell-clock-2026-11-20/%j.out
# The 2026-11-20 diagnostic of dell's slow one-thread runs (job 546; see
# bench/cluster-cpu-scaling/BENCHMARK.md). From the repository root, with
# ./tree_cpu built: sbatch bench/tools/cluster/dell-clock/diag.sh
set -u
R=$PWD D=$PWD/bench/cluster-cpu-scaling/dell-clock-2026-11-20
T=$R/tree_cpu; X=$R/bench/tools/data/higgs/train.f32
P=/sys/devices/system/cpu
echo "== info $(hostname) $(date -Is)"
for f in scaling_driver scaling_governor energy_performance_preference energy_performance_available_preferences scaling_max_freq; do echo "$f: $(cat $P/cpu0/cpufreq/$f 2>&1)"; done
ls -l $P/cpu0/cpufreq/energy_performance_preference $P/cpu0/cpufreq/scaling_governor
[ -w $P/cpu0/cpufreq/energy_performance_preference ] && echo "EPP writable" || echo "EPP not writable"
for f in status no_turbo min_perf_pct max_perf_pct hwp_dynamic_boost; do echo "intel_pstate/$f: $(cat $P/intel_pstate/$f 2>&1)"; done
echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing)"
echo "platform_profile: $(cat /sys/firmware/acpi/platform_profile 2>&1)"
for d in $P/intel_uncore_frequency/*/; do echo "uncore $d: min $(cat $d/min_freq_khz) max $(cat $d/max_freq_khz) init_max $(cat $d/initial_max_freq_khz 2>/dev/null) cur $(cat $d/current_freq_khz)"; done
for d in /sys/class/powercap/intel-rapl:?; do echo "rapl $d $(cat $d/name): $(cat $d/constraint_0_name 2>&1)=$(cat $d/constraint_0_power_limit_uw 2>&1) $(cat $d/constraint_1_name 2>&1)=$(cat $d/constraint_1_power_limit_uw 2>&1) energy_readable=$([ -r $d/energy_uj ] && echo y || echo n)"; done
for h in /sys/class/hwmon/hwmon*; do echo "hwmon $h $(cat $h/name)"; done
sudo -n true 2>/dev/null && echo "SUDO yes" || echo "SUDO no"
numactl -H 2>&1 | head -12
for c in 0 1 2 3; do echo "cpu$c package $(cat $P/cpu$c/topology/physical_package_id) node $(ls -d $P/cpu$c/node* | sed 's/.*node//')"; done
RR=$(seq -s, 0 47)
COMPACT=$(python3 -c "
import glob,re
c=[]
for p in glob.glob('$P/cpu[0-9]*'):
    i=int(re.search(r'cpu(\d+)$',p).group(1)); pk=int(open(p+'/topology/physical_package_id').read()); nd=int(glob.glob(p+'/node*')[0].rsplit('node',1)[1])
    c.append((pk,nd,i))
print(','.join(str(i) for _,_,i in sorted(c)))")
echo "compact order: $COMPACT"
python3 $R/bench/tools/cluster/dell-clock/sampler.py 0.2 > $D/samples-$SLURM_JOB_ID.csv &
SP=$!
vm() { awk '/^(pgmigrate_success|numa_hit|numa_miss|numa_local|numa_other) /{printf "%s=%s ",$1,$2}' /proc/vmstat; }
run() { # phase threads order
  local cpus; cpus=$(echo "$3" | cut -d, -f1-$2)
  local v0; v0=$(awk '/^pgmigrate_success /{print $2}' /proc/vmstat)
  local s; s=$(date +%s.%N)
  local out; out=$(taskset -c $cpus $T --multicore --threads $2 --cart -d 10 $X 2>&1)
  local e; e=$(date +%s.%N)
  local v1; v1=$(awk '/^pgmigrate_success /{print $2}' /proc/vmstat)
  local ms; ms=$(echo "$out" | grep -oP 'train total\s+\K[\d.]+')
  echo "RUN phase=$1 threads=$2 order=$4 start=$s end=$e train_ms=$ms pgmigrate=$((v1-v0))"
}
echo "== A: interleaved like the scaling job"
for i in $(seq 1 20); do run A 1 $RR rr; run A 48 $RR rr; run A 8 $RR rr; done
echo "== B: one-thread runs back to back"
for i in $(seq 1 10); do run B 1 $RR rr; done
echo "== C: placement"
for i in 1 2 3 4; do for t in 2 4 8 12 24; do run C $t $RR rr; run C $t $COMPACT compact; done; done
kill $SP
echo "== done $(date -Is)"
