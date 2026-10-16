# ./tree on the mordor3 cluster

Scripts that build, prepare and benchmark ./tree on the faculty's mordor3
cluster (Politechnika Białostocka, SLURM), and tune it there. Results:

- [`bench/results/cluster-higgs/`](../results/cluster-higgs/BENCHMARK.md): HIGGS (10.5M rows)
- [`bench/results/cluster-airlines/`](../results/cluster-airlines/BENCHMARK.md): airline (115M rows), pruned and (in `noprunning/`) unpruned trees
- [`bench/results/cluster-cpu-scaling/`](../results/cluster-cpu-scaling/BENCHMARK.md): HIGGS with 1 to 48 threads on the two CPU nodes
- `bench/results/cluster-ecbdl/`: ECBDL'14 (631 features): a 6M-row sample on all four nodes, the full 32M rows on the CPU nodes

Four nodes, two of each kind. **The GPU nodes (hopper, ampere) run CUDA
only, the CPU nodes (epyc1, dell1) parallel only:** a GPU node's CPU is
there to feed the GPU and is not benchmarked on its own. turing (RTX 2080
Ti) is not used.

| Node | CPU | RAM | GPU (GPU 0 used) |
|---|---|---|---|
| hopper | 2× Xeon Gold 6444Y (Sapphire Rapids), 32 cores | 1 TB DDR5-4800 | 4× H100 PCIe 80 GB |
| ampere | 2× EPYC 7402 (Zen 2), 48 cores | 1 TB DDR4-3200 | 2× A100 80 GB + 2× A100 40 GB PCIe (GPU 0: 80 GB) |
| epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – |
| dell1 | 2× Xeon Gold 6542Y (Emerald Rapids), 48 cores | 1 TB DDR5-5200 | – |

## Run it

On mordor3, in a clone of this repository (`~/dt`). Nothing heavy runs on the
login node: every step is a SLURM job.

```bash
# once: the Python venv (python3-venv is missing there; virtualenv is installed)
virtualenv -p python3 bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt
# once: the downloads, see "Data" below, into ~/downloads
sbatch --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)" \
    bench/cluster/build.sh       # on hopper: ./tree for sm_75 + sm_80 + sm_90, ./tree_cpu, rusage
sbatch bench/cluster/prepare.sh  # on a dell node: the datasets (checked by SHA-256), HIGGS prepared
sbatch --export=ALL,PREPARE=airline -t 02:00:00 -o bench/cluster/prepare-airline.log \
    bench/cluster/prepare.sh airline
bench/cluster/submit.sh -m 3               # HIGGS: four jobs, one per node, released together
bench/cluster/submit.sh -m 3 -d airline    # airline (or name single runs, e.g. cuda-hopper-airline)
bench/cluster/scaling.sh -m 3              # HIGGS with 1, 2, 4, 8, 12, ... 48 threads on epyc1 and dell1
```

ECBDL'14, each case once. Preparing it needs about 350 GB of RAM, so it runs
on ampere (1 TB); the benchmarks wait for it (`-a`):

```bash
ln -s /home/Datasets/ECBDL ~/downloads/ecbdl
prep=$(sbatch --parsable -p ampere -w ampere --exclusive -c 48 -t 08:00:00 \
    --export=ALL,PREPARE="ecbdl ecbdl_6m" -o bench/cluster/prepare-ecbdl.log \
    bench/cluster/prepare.sh ecbdl)
bench/cluster/submit.sh -d ecbdl_6m -a "$prep"   # all four nodes
bench/cluster/submit.sh -d ecbdl -a "$prep"      # full: epyc1 and dell1 only
```

Then copy `bench/results/cluster-*/` back and draw the charts:

```bash
for d in higgs:higgs airlines:airline; do
  dir=bench/results/cluster-${d%%:*} data=${d##*:}
  bench/.venv/bin/python bench/machines_chart.py --protocols cart_alpha_nodepth,c45 \
      $dir/cuda-hopper-$data $dir/cuda-ampere-$data $dir/cpu-epyc-$data $dir/cpu-dell-$data \
      --out $dir/chart.png
done
for node in epyc dell; do  # the scaling charts, one per node
  bench/.venv/bin/python bench/scaling_chart.py bench/results/cluster-cpu-scaling/cpu-$node-higgs \
      --out bench/results/cluster-cpu-scaling/chart-$node.png
done
```

The unpruned airline run and its chart:

```bash
bench/cluster/submit.sh -m 3 -d airline -p cart_full_nodepth,c45_unpruned -o cluster-airlines/noprunning
dir=bench/results/cluster-airlines/noprunning
bench/.venv/bin/python bench/machines_chart.py --protocols cart_full_nodepth,c45_unpruned \
    $dir/cuda-hopper-airline $dir/cuda-ampere-airline $dir/cpu-epyc-airline $dir/cpu-dell-airline \
    --out bench/results/cluster-airlines/chart-noprunning.png
```

- **`build.sh`** runs on hopper because the CPU nodes have no CUDA toolkit. It
  builds with `NVCC=/usr/local/cuda/bin/nvcc` and `-gencode` for sm_75, sm_80
  and sm_90, so one binary serves the A100, the H100 and turing's RTX 2080 Ti, and
  rebuilds the CUDA object when that list changes. Log: `build.log`.
- **`submit.sh`** queues one job per node (`job.sh`) with `--exclusive` and
  `--hold`, then releases them together, so on idle nodes they start in the
  same second. Each job runs
  `bench/run.py <dataset> --protocols cart_alpha_nodepth,c45 --impls tree|tree_cuda --threads all --cpus 0-<cores-1> --reps <-m> --run-id cluster-<…>/<run> --train-accuracy-only`
  and `report.py`: `tree_cuda` on the GPU nodes, `tree` (parallel) on the
  CPU nodes. Its log is `<run>/run.log`.
- **`scaling.sh`** queues the same job on epyc1 and dell1 with the thread list
  `1,2,4,8,12,…,48` (1 thread: `--serial`); `run.log` records the CPU order
  the threads take (`lscpu`).
- **`tune.sh`** runs a list of ./tree command lines on one node, interleaved
  and repeated, and prints ./tree's phase times with the CPU time, page faults
  and context switches of every run. The default settings were tuned
  with it ([docs/PERFORMANCE.md](../../docs/PERFORMANCE.md)).

## Settings specific to the cluster

The data and measurements are those of the other benchmarks
([bench/README.md](../README.md)), with every CPU of each node, and two
differences:

- **No depth limit for CART** (`cart_alpha_nodepth`, `cart_full_nodepth`).
  The other benchmarks cap CART at depth 30 because rpart cannot grow deeper
  and they compare ./tree with rpart; here only ./tree runs, and C4.5 has no
  cap either.
- **Training accuracy only** (`run.py --train-accuracy-only`): ./tree's own,
  printed after its timers. The test rows are not classified: predicting them
  in Python from the dumped tree took minutes per run on airline.


| Backend | Command | Threads |
|---|---|---|
| Parallel (epyc1, dell1) | `./tree_cpu --parallel --threads N` | N = every CPU (48) |
| CUDA (hopper, ampere) | `./tree --cuda --threads N` | GPU 0, plus N CPU threads (hopper 32, ampere 48) for the small subtrees and pruning |

- **Whole nodes.** SLURM on mordor3 hands out cores only: it tracks no GPUs
  (no GRES) and no memory. Every job therefore takes its node with
  `--exclusive`, so no other job shares its cores, memory bandwidth or GPUs.
  `run.log` records the load average and the GPU processes at the start.
- **One GPU.** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`: ./tree
  uses GPU 0 in nvidia-smi's order. That is the GPU whose memory `run.py`
  samples (`nvidia-smi --id=0`); on ampere it is an A100 80 GB. The other
  GPUs sit idle.
- **CPU governors are as the admins set them.** epyc1 `performance`, dell1
  `powersave` (intel_pstate, which still boosts), hopper
  `schedutil`, ampere `ondemand`. A user can't change them; `machine.json`
  records them. All four dell nodes are set alike (checked on 2026-10-14:
  `powersave` on all 48 CPUs, intel_pstate, energy-performance preference
  `balance_performance`, turbo on), so dell1 stands for all of them. With
  intel_pstate, `powersave` is the dynamic governor that still reaches turbo,
  not acpi-cpufreq's `powersave`, which holds the lowest frequency.
- **Hopper's CPU times drift** over a day by up to 40% (the same binary: CART
  parallel 2.2 s at 12:45, 3.2 s at 15:15, with no other process on the node).
  Comparisons of two versions there are therefore made in one job, the runs
  interleaved (`tune.sh`).
- **Hopper has no git.** `submit.sh` records the commit on the login node, and
  `run.py` takes it from `TREE_GIT_DESCRIBE`.
- **Data over NFS.** /home is mordor3's RAID, NFS-mounted on every node; SLURM
  copies nothing. Reading the CSV is not timed, so this only lengthens the
  wall time.
- **RAM in the chart** is the kernel's MemTotal with the module type and
  speed. udev's DMI list, the usual source, is cut short on these boards
  (11–15 of 16 modules).

## Data

The cluster gets its data from the internet, not from the desktop (whose
upload is slow). `make_datasets.py` (run by `prepare.sh`) rebuilds
`datasets/*.csv` byte for byte as the desktop copies, makes `airline.csv`, and
stops on any SHA-256 difference:

| File | Source | SHA-256 (= desktop; airline: first build) |
|---|---|---|
| `covertype.csv` | UCI 31 `covertype.zip` (same rows and order; `Id`, header and class names added) | `fd7a387d…` |
| `higgs.csv` | OpenML 45570 parquet: UCI HIGGS, same rows and order. UCI's own zip downloads at 0.5 MB/s, OpenML's at 19 MB/s. The parquet's doubles are slightly off the float32 values, so they are rounded back to float32 and printed as UCI does (`%.18e`, CRLF line ends, `Id`, label last as signal/background) | `71f566a2…` |
| `supersymmetry.csv` | UCI 279 `susy.zip`: the same float32 values and labels, but the desktop copy's rows are shuffled (no common seed reproduces the order) and its doubles are UCI's text read by an unknown parser, 0–6 ulps off. Two files made from the desktop copy carry what is not public: `susy_order.u64` (40 MB, a hash of every row in the desktop's order) and `susy_delta.i8` (90 MB, every value's offset in float64 ulps; 21 MB zstd) | `85ecf105…` |
| `airline.csv` | gbm-bench's airline dataset (also in the XGBoost GPU paper): US flights from October 1987 to April 2008, `airline_14col.data.bz2` from kt.ijs.si (1.09 GB, SHA-256 `1f13460f…`). 115,069,017 rows, 13 integer features; the carrier and the airports become codes in sorted order, as in gbm-bench, and the label is `ArrDelay > 0` (`delayed` 47.5%, `on_time` 52.5%). Rows in file (date) order, so `datasets.toml` takes a random 500,000-row test set. There is no desktop copy: the hash is that of its first build. It takes about 3 minutes, and `prepare.py` 5 more | `25542603…` |

Airline has 10.5× HIGGS's rows and 4.9× its values (1.49 G against 0.31 G;
6.0 GB as float32 against 1.2 GB). Its CSV is smaller than HIGGS's only
because its values are short integers. It is cluster-only: the CUDA backend
needs 23.6 GB of GPU memory for it, more than the laptop's 8 GB or a 2080 Ti's 11 GB.
`ActualElapsedTime` partly gives the delay away, so its accuracy says little;
it is there to compare speed, as in gbm-bench.

`ecbdl.csv` is ECBDL'14 (protein contact-map prediction, the GECCO 2014 big
data competition) from the ARFF in mordor3's `/home/Datasets/ECBDL` (60.8 GB):
31,992,921 rows, 631 features, label 0/1 (about 2% contacts). The 539 real
attributes become float32; the 92 nominal ones (predicted secondary structure
H/E/C, solvent accessibility 0–4, X where the window runs past the chain) become
integer codes in sorted order, as airline's airports. Its 20.2 G values are 65×
HIGGS's (81 GB as float32). `datasets.toml` sets 100,000 random rows aside as
the test set and writes no YaDT files. The CUDA backend needs 16 bytes per value
(20 for C4.5, which keeps the values for its pruning), about 320 GB for the full
set, so the GPUs get `ecbdl_6m`: a random 6M of the training rows (3.8 G values,
about 76 GB with C4.5's copy). The pre-converted `.bin` files under
`/home/Datasets/IMM` lack the label and the nominal attributes.

HIGGS's prepared files (`bench/data/higgs/`) have the SHA-256 recorded in
`plan.json` of `pc-legion-higgs/cpu-pc-higgs` and `pc-legion-higgs/cuda-legion-higgs` (desktop and laptop).
Logs: `prepare-higgs.log`, `prepare-airline.log`.
