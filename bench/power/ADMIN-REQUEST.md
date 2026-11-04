# Request: read-only access to CPU and PSU power sensors on mordor3

For a research paper we measure the energy used by our programs. This needs
read access to the CPU and PSU power sensors for normal users. The setup
below needs root once; afterwards, no root or sudo is required. All changes
are read-only and give users no control over the hardware.

| Node | CPU power | GPU power | PSU power |
|---|---|---|---|
| dell1–4, epyc1–2 | Step 1: make RAPL counters readable | – | Available |
| ampere | Available (via perf) | Available | Step 2: publish the IPMI reading |
| hopper | Step 1: make RAPL counters readable | Available | Step 2: publish the IPMI reading |

## Step 1: CPU power (dell1–4, epyc1–2, hopper)

The RAPL energy counters (`/sys/class/powercap/intel-rapl:*/energy_uj`) are
readable by root only. Please make them world-readable, persistently across
reboots:

```bash
echo 'ACTION=="add", SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chmod 0444 /sys%p/energy_uj"' \
  > /etc/udev/rules.d/99-rapl-read.rules
udevadm control --reload
chmod 0444 /sys/class/powercap/intel-rapl:*/energy_uj
```

## Step 2: PSU power (ampere, hopper)

These servers expose PSU power only through the BMC (`ipmitool`), which
requires root. Instead of opening `/dev/ipmi0`, please run a small service
that publishes the reading once per second in a world-readable file. (Not
needed on hopper if `grep power_meter /sys/class/hwmon/*/name` finds an ACPI
power meter, as on the Dell servers.)

```bash
cat > /usr/local/sbin/psu-poll <<'EOF'
#!/bin/sh
mkdir -p /run/power
while true; do
  w=$(ipmitool dcmi power reading | awk '/Instantaneous/ {print $4}')
  echo "$(date +%s.%N) $w" > /run/power/psu_W.tmp && mv /run/power/psu_W.tmp /run/power/psu_W
  sleep 1
done
EOF
chmod 0755 /usr/local/sbin/psu-poll

cat > /etc/systemd/system/psu-poll.service <<'EOF'
[Unit]
Description=Publish PSU power for users in /run/power/psu_W
[Service]
ExecStart=/usr/local/sbin/psu-poll
Restart=always
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload && systemctl enable --now psu-poll
```

## Verification (as a normal user)

```bash
cat /sys/class/powercap/intel-rapl:0/energy_uj   # Step 1: prints a number
cat /run/power/psu_W                             # Step 2: prints "<unix time> <watts>"
```
