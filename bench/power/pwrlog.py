#!/usr/bin/env python3
"""Sample every power source a normal user can read, every DT seconds, until
killed (SIGTERM or SIGINT), and write CSV.

  python3 bench/power/pwrlog.py [DT=0.1] [OUT=stdout]

Columns:
  t          Unix time of the sample
  cpu_busy   fraction of all CPUs busy since the previous sample (/proc/stat)
  sys_W      ACPI power meter (hwmon "power_meter", the Dell nodes): the whole
             server's PSU input power as the iDRAC reports it, a 2 s average,
             refreshed every ~2 s (dell) or 4-6 s (epyc)
  psu_W      /run/power/psu_W (ampere, hopper: an admin service writes the BMC's
             DCMI reading there every second; on ampere the BMC refreshes the
             value only about once a minute)
  rapl_*_uJ  RAPL energy counters, where they are readable (since 2026-11-06 on
             ampere and hopper; they wrap at max_energy_range_uj, which
             power_bench.sh writes to machine.txt)
  gpu<i>_W   NVML board power of GPU i
  gpu<i>_mJ  NVML total-energy counter of GPU i (millijoules since driver load)
  t_end      Unix time after the last read of the sample. NVML calls can block
             for ~2 s (on ampere, each time a GPU tree ends), so a counter may be
             read up to 2 s after `t`: analyze.py times the counters with t_end
"""
import ctypes
import glob
import os
import signal
import sys
import time

dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout
cols, readers = ["t"], []


def cpu_times():
    fields = open("/proc/stat").readline().split()[1:]
    values = [int(v) for v in fields]
    idle = values[3] + values[4]  # idle + iowait
    return sum(values[:8]), idle


last = [cpu_times()]


def cpu_busy():
    total, idle = cpu_times()
    dtotal, didle = total - last[0][0], idle - last[0][1]
    last[0] = (total, idle)
    return round(1 - didle / dtotal, 4) if dtotal > 0 else 0.0


cols.append("cpu_busy")
readers.append(cpu_busy)
for h in glob.glob("/sys/class/hwmon/hwmon*"):
    if open(h + "/name").read().strip() == "power_meter":
        p = h + "/power1_average"
        cols.append("sys_W")
        readers.append(lambda p=p: int(open(p).read()) / 1e6)
PSU_FILE = "/run/power/psu_W"


def psu():
    try:
        return float(open(PSU_FILE).read().split()[1])
    except (OSError, IndexError, ValueError):  # mid-update or the service is down
        return float("nan")


if os.path.exists(PSU_FILE):
    cols.append("psu_W")
    readers.append(psu)
for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
    f = d + "/energy_uj"
    try:
        open(f).read()
    except OSError:
        continue
    cols.append("rapl_%s_%s_uJ" % (os.path.basename(d), open(d + "/name").read().strip()))
    readers.append(lambda f=f: int(open(f).read()))
try:
    nv = ctypes.CDLL("libnvidia-ml.so.1")
    nv.nvmlInit_v2()
    n = ctypes.c_uint()
    nv.nvmlDeviceGetCount_v2(ctypes.byref(n))
    for i in range(n.value):
        hd = ctypes.c_void_p()
        nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(hd))

        def power(hd=hd):
            v = ctypes.c_uint()
            nv.nvmlDeviceGetPowerUsage(hd, ctypes.byref(v))
            return v.value / 1000

        def energy(hd=hd):
            v = ctypes.c_ulonglong()
            nv.nvmlDeviceGetTotalEnergyConsumption(hd, ctypes.byref(v))
            return v.value

        cols += ["gpu%d_W" % i, "gpu%d_mJ" % i]
        readers += [power, energy]
except OSError:
    pass

signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
print(",".join(cols + ["t_end"]), file=out, flush=True)
nxt = time.time()
while True:
    t = time.time()
    values = [str(r()) for r in readers]
    print(",".join(["%.3f" % t] + values + ["%.3f" % time.time()]), file=out, flush=True)
    nxt += dt
    time.sleep(max(0.0, nxt - time.time()))
