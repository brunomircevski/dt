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
  rapl_*_uJ  RAPL energy counters, only where they are readable (root-only on
             mordor3; on ampere bench/power/power_bench.sh reads RAPL with perf)
  gpu<i>_W   NVML board power of GPU i
  gpu<i>_mJ  NVML total-energy counter of GPU i (millijoules since driver load)
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
print(",".join(cols), file=out, flush=True)
nxt = time.time()
while True:
    print(",".join(["%.3f" % time.time()] + [str(r()) for r in readers]), file=out, flush=True)
    nxt += dt
    time.sleep(max(0.0, nxt - time.time()))
