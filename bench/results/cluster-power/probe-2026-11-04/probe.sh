#!/bin/bash
# Power-measurement capability probe (read-only)
echo "===== $(hostname) $(date -Is) kernel $(uname -r)"
echo "--- DMI"; for f in sys_vendor product_name board_vendor board_name bios_version; do printf "%s: " $f; cat /sys/class/dmi/id/$f 2>&1; done
echo "--- CPU"; lscpu | egrep 'Model name|Socket|Thread|Core|NUMA node\(s\)'
echo "--- id"; id
echo "--- powercap"; ls -la /sys/class/powercap/ 2>&1
for d in /sys/class/powercap/*/; do [ -f $d/name ] || continue; printf "%s name=%s " "$d" "$(cat $d/name)"; ls -la $d/energy_uj 2>&1 | awk '{printf "%s %s %s ",$1,$3,$4}'; cat $d/energy_uj 2>&1 | head -c 80; echo; cat $d/max_energy_range_uj 2>/dev/null; done
echo "--- perf"; cat /proc/sys/kernel/perf_event_paranoid; ls /sys/bus/event_source/devices/; ls /sys/bus/event_source/devices/power/events/ 2>&1; which perf 2>&1
perf stat -a -e power/energy-pkg/ -e power/energy-ram/ -e power/energy-cores/ sleep 1 2>&1 | tail -12
echo "--- msr"; ls -la /dev/cpu/0/msr 2>&1; lsmod | egrep -i 'msr|rapl|amd_energy|zenpower|power_meter|ipmi|acpi_ipmi|k10temp|coretemp|intel_uncore' 
echo "--- hwmon"; for h in /sys/class/hwmon/hwmon*; do printf "%s %s: " $h "$(cat $h/name 2>/dev/null)"; ls $h | egrep 'power|energy|curr|in[0-9]_' | tr '\n' ' '; echo; for p in $h/power*_input $h/power*_average $h/energy*_input; do [ -e $p ] && { printf "   %s = " $p; cat $p 2>&1; }; done; done
echo "--- acpi power_meter"; ls /sys/bus/acpi/drivers/power_meter/ 2>&1; ls /sys/bus/acpi/devices/ | grep -i ACPI000D 2>&1
echo "--- ipmi"; ls -la /dev/ipmi* /dev/ipmidev* 2>&1; which ipmitool ipmi-dcmi ipmi-sensors 2>&1
ipmitool dcmi power reading 2>&1 | head -12
echo "--- tools"; for t in nvidia-smi dcgmi likwid-powermeter likwid-perfctr turbostat powerstat powertop scaphandre rocm-smi amd-smi uProf AMDuProfCLI s-tui sensors cpupower python3 R java; do printf "%s: " $t; which $t 2>&1 | head -1; done
echo "--- nvidia"; if which nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index,name,power.draw,power.limit,enforced.power.limit,memory.used,utilization.gpu --format=csv; nvidia-smi -q -d POWER | head -40; nvidia-smi --version; fi
echo "--- systemd/dcgm/services"; systemctl list-units --type=service --no-pager 2>/dev/null | egrep -i 'ipmi|dcgm|nvidia|power|bmc|redfish|prometheus|node_exporter|collect' 
echo "--- net (BMC hint)"; ip -br addr 2>/dev/null
echo "--- governor"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>&1
echo "--- dimm count"; ls /sys/devices/system/edac/mc/ 2>&1; ls /sys/devices/system/edac/mc/mc*/ 2>/dev/null | grep -c dimm
