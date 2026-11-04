# Request: let users read CPU and PSU power on mordor3

We measure the energy that our programs use (for a paper). We need to read
two kinds of power sensors as a normal user. Each step below needs root
**once**; after that, nothing needs root or sudo. Every step is read-only:
users get no control over the hardware.

| Node | CPU power (step 1) | GPU power | PSU power (step 2) |
|---|---|---|---|
| dell1–4, epyc1–2 | **needed** | – | already works, nothing to do |
| ampere | already works (perf_event_paranoid=0), do step 1 anyway | already works | **needed** |
| hopper | **needed** | already works | **needed** (unless the check in step 2 finds a power meter) |

## Step 1: CPU power (RAPL) readable by users — all nodes

The RAPL energy counters in `/sys/class/powercap/intel-rapl:*/energy_uj`
are root-only (mode 0400). Please make them world-readable at every boot
with a udev rule. On AMD nodes the same `intel-rapl` files exist.

```bash
cat > /etc/udev/rules.d/99-rapl-read.rules <<'EOF'
ACTION=="add", SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chmod 0444 /sys%p/energy_uj"
EOF
udevadm control --reload
chmod 0444 /sys/class/powercap/intel-rapl:*/energy_uj   # now, without a reboot
```

Background: the counters were made root-only because of the PLATYPUS side
channel (CVE-2020-8694). Jobs on these nodes are exclusive, so we think this
is acceptable. If you prefer to allow only a group (for example `power`,
with user `bmircevski` in it), use this rule instead, and `chgrp`/`chmod 0440`
for the immediate change:

```
ACTION=="add", SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/sh -c 'chgrp power /sys%p/energy_uj; chmod 0440 /sys%p/energy_uj'"
```

## Step 2: PSU power on ampere and hopper

The Dell servers already show their PSU power to users (ACPI power meter).
ampere (Supermicro) has no such meter. Its PSU power can only be read from
the BMC with `ipmitool`, which needs `/dev/ipmi0` (root-only). Please do
**not** make `/dev/ipmi0` readable: it also allows powering the machine off.
Instead, please run a small root service that reads the PSU power every
second and writes it to a file anyone can read.

First, a check on hopper: if this prints a path, hopper has a power meter
and needs no service:

```bash
grep -l power_meter /sys/class/hwmon/hwmon*/name
```

Check that the BMC answers (install `ipmitool` first if it is missing):

```bash
ipmitool dcmi power reading     # should print "Instantaneous power reading: ... Watts"
```

Then install the service:

```bash
cat > /usr/local/sbin/psu-poll <<'EOF'
#!/bin/sh
# Publish the PSU input power for users: /run/power/psu_W holds "<unix time> <watts>".
mkdir -p /run/power
chmod 0755 /run/power
while true; do
  w=$(ipmitool dcmi power reading 2>/dev/null | awk '/Instantaneous power reading/ {print $4}')
  echo "$(date +%s.%N) $w" > /run/power/psu_W.tmp
  chmod 0644 /run/power/psu_W.tmp
  mv /run/power/psu_W.tmp /run/power/psu_W
  sleep 1
done
EOF
chmod 0755 /usr/local/sbin/psu-poll

cat > /etc/systemd/system/psu-poll.service <<'EOF'
[Unit]
Description=Publish PSU power reading for users (/run/power/psu_W)

[Service]
ExecStart=/usr/local/sbin/psu-poll
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now psu-poll
```

If `ipmitool dcmi power reading` does not work on a node, please send us the
output of `ipmitool sdr elist | grep -i -E 'watt|pw|psu'`, and we will adapt
the script to read the PSU sensors instead.

## How we check it worked (as a normal user)

```bash
cat /sys/class/powercap/intel-rapl:0/energy_uj   # step 1: prints a number, no "Permission denied"
cat /run/power/psu_W                             # step 2 (ampere, hopper): prints "<time> <watts>"
```

GPU power needs nothing: NVML (`nvidia-smi`) already works for users.
