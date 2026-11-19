# Sourced by the tuning scripts (ab.sh, tune.sh) on a GPU node: picks the GPU as
# bench/tools/cluster/job.sh does (hopper GPU 2, ampere GPU 0; elsewhere a GPU with the
# most memory, the first on a tie; CUDA_VISIBLE_DEVICES if already set) and stops unless it is an 80 GB card
# (ampere: GPUs 0-1 A100 80 GB, GPUs 2-3 A100 40 GB). GPU_MIN_MIB: default 80000.
# No GPU (nvidia-smi missing): nothing happens.
if command -v nvidia-smi > /dev/null; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID  # CUDA's device numbers = nvidia-smi's
  case "$(hostname -s)" in hopper) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} ;; ampere) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ;; esac
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=memory.total,index --format=csv,noheader,nounits |
      sort -s -t, -k1,1nr | head -1 | cut -d, -f2 | tr -d ' ')"
    export CUDA_VISIBLE_DEVICES
  fi
  gpu_mib="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=memory.total --format=csv,noheader,nounits)"
  [ "${gpu_mib:-0}" -ge "${GPU_MIN_MIB:-80000}" ] ||
    { echo "GPU $CUDA_VISIBLE_DEVICES has ${gpu_mib:-?} MiB, under ${GPU_MIN_MIB:-80000}: not an 80 GB card" >&2; exit 1; }
  echo "GPU $CUDA_VISIBLE_DEVICES: $(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total --format=csv,noheader)"
fi
