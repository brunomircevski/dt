#!/usr/bin/env bash
# The end of every job (job.sh, build.sh: trap ... EXIT), even a failed or
# timed-out one: leave the node as we found it.
#
#   cleanup.sh <job temp folder> [data folder]
#
# - removes the job's temp folder (job.sh points TMPDIR there: run.py's
#   scratch, Python's temp files, nvcc's tmpxft_*). /tmp on the nodes is
#   tmpfs, so leftovers there hold RAM.
# - drops the datasets from the node's page cache (they stay only on the login
#   node's disk, read over NFS): cached CSVs piled up on one NUMA node (236 GiB
#   of epyc1's node 1 on 2026-10-20) and slowed the next jobs. run.py
#   --drop-cache does the same before every run; this catches killed runs.
#   dd iflag=nocache count=0 is posix_fadvise DONTNEED on the whole file.
dir="${1:-}" data="${2:-}"
case "$dir" in /tmp/dt-*) rm -rf -- "$dir" ;; esac  # only a folder job.sh made
[ -z "$data" ] || find -L "$data" -type f -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null
true
