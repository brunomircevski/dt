#!/usr/bin/env python3
# every DT s: t, cpu0 MHz, mean/min/max MHz over all CPUs, package temps, RAPL package energy, uncore MHz
import glob, sys, time
dt = float(sys.argv[1])
cpus = sorted(glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq'), key=lambda p: int(p.split('cpu')[2].split('/')[0]))
temps = []
for h in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
    try: name = open(h + '/name').read().strip()
    except OSError: continue
    if name == 'coretemp':
        for f in sorted(glob.glob(h + '/temp*_label')):
            if open(f).read().startswith('Package'): temps.append(f.replace('_label', '_input'))
rapl = [d + '/energy_uj' for d in sorted(glob.glob('/sys/class/powercap/intel-rapl:[0-9]'))]
rapl = [f for f in rapl if (lambda f: (open(f).read(), True)[1] if True else False)(f)] if all(__import__('os').access(f, 4) for f in rapl) else []
unc = [u for u in sorted(glob.glob('/sys/devices/system/cpu/intel_uncore_frequency/*/current_freq_khz')) if __import__('os').access(u, 4)]
print('t,cpu0,mean,min,max,' + ','.join(f'temp{i}' for i in range(len(temps))) + ',' + ','.join(f'rapl{i}_uJ' for i in range(len(rapl))) + ',' + ','.join(f'unc{i}' for i in range(len(unc))), flush=True)
while True:
    f = [int(open(c).read()) // 1000 for c in cpus]
    row = [f'{time.time():.2f}', f[0], sum(f) // len(f), min(f), max(f)]
    row += [int(open(t).read()) // 1000 for t in temps] + [int(open(r).read()) for r in rapl] + [int(open(u).read()) // 1000 for u in unc]
    print(','.join(map(str, row)), flush=True)
    time.sleep(dt)
