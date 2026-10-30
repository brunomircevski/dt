# ./tree on the mordor3 cluster

Scripts that build, prepare and benchmark ./tree on the faculty's mordor3
cluster (Politechnika Białostocka, SLURM), and tune it there. Results:

- The test-set benchmarks ([below](#test-set-benchmarks)): ./tree trained on each
  dataset's training rows and scored on its test rows, depth ≤ 10:
  [`cluster-airline/`](../results/cluster-airline/BENCHMARK.md),
  [`cluster-higgs/`](../results/cluster-higgs/BENCHMARK.md),
  [`cluster-susy/`](../results/cluster-susy/BENCHMARK.md),
  [`cluster-ecbdl/`](../results/cluster-ecbdl/BENCHMARK.md)
- [`bench/results/cluster-cpu-scaling/`](../results/cluster-cpu-scaling/BENCHMARK.md): HIGGS and MNIST8M with 1 to 48 threads on the two CPU nodes
- [`bench/results/cluster-chess/`](../results/cluster-chess/BENCHMARK.md): a 3×3 checkerboard (2 features) at 1M, 10M, 100M and 1000M rows on all four nodes and dell2, with one thread on epyc1 and dell1, and `--gpu-only` on hopper and ampere; unpruned CART and C4.5

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
# once: the downloads, see "Data" below, into bench/data/downloads
sbatch --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)" \
    bench/cluster/build.sh       # on hopper: ./tree for sm_75 + sm_80 + sm_90, ./tree_cpu, rusage
sbatch bench/cluster/prepare.sh  # on a dell node: the datasets (checked by SHA-256), HIGGS prepared
sbatch --export=ALL,PREPARE=airline -t 02:00:00 -o bench/cluster/prepare-airline.log \
    bench/cluster/prepare.sh airline
bench/cluster/submit.sh -m 3               # HIGGS: four jobs, one per node, released together
bench/cluster/submit.sh -m 3 -d airline    # airline (or name single runs, e.g. cuda-hopper-airline)
bench/cluster/scaling.sh -m 3              # HIGGS with 1, 2, 4, 8, 12, ... 48 threads on epyc1 and dell1
bench/cluster/scaling.sh -m 3 -d airline   # the same on airline
bench/cluster/scaling.sh -m 10 -d mnist8m  # the 2026-10-29/30 runs: submit-main-2026-10-29.sh
# one socket of epyc1 and dell1: all its CPUs, the memory of its NUMA nodes
bench/cluster/submit.sh -m 3 socket-epyc-higgs socket-dell-higgs
bench/cluster/submit.sh -m 3 -d airline socket-epyc-airline socket-dell-airline
# --simple, --serial, parallel, scikit-learn, J48 at depth 10, one job each (cluster-cpu-vs-others/BENCHMARK.md)
bench/cluster/submit.sh -m 1 -e -d covertype -o cluster-cpu-vs-others -w epyc2 vs-epyc-covertype-serial-cart vs-epyc-covertype-j48-c45
```

ECBDL'14, each case once. Preparing it needs about 350 GB of RAM, so it runs
on ampere (1 TB); the benchmarks wait for it (`-a`):

```bash
ln -s /home/Datasets/ECBDL bench/data/downloads/ecbdl
prep=$(sbatch --parsable -p ampere -w ampere --exclusive -c 48 -t 08:00:00 \
    --export=ALL,PREPARE="ecbdl ecbdl_15m" -o bench/cluster/prepare-ecbdl.log \
    bench/cluster/prepare.sh ecbdl)
bench/cluster/submit.sh -d ecbdl_15m -a "$prep"  # all four nodes
bench/cluster/submit.sh -d ecbdl -a "$prep"      # full: epyc1 and dell1 only
bench/cluster/submit.sh -d ecbdl_15m socket-epyc-ecbdl_15m socket-dell-ecbdl_15m  # one socket
```

Then copy `bench/results/cluster-*/` back and draw the charts:

```bash
dir=bench/results/cluster-higgs  # airline has its own, pruned and unpruned in one image, below
bench/.venv/bin/python bench/machines_chart.py --protocols cart_alpha_nodepth,c45 \
    $dir/cuda-hopper-higgs $dir/cuda-ampere-higgs $dir/cpu-epyc-higgs $dir/cpu-dell-higgs \
    $dir/socket-epyc-higgs $dir/socket-dell-higgs --out $dir/chart.png
s=bench/results/cluster-cpu-scaling
t=$(mktemp -d)  # one chart per node and dataset, only to build the grid
for d in higgs mnist8m; do for node in epyc dell; do
  bench/.venv/bin/python bench/scaling_chart.py $s/cpu-$node-$d \
      --socket bench/results/cluster-$d/socket-$node-$d --out $t/$node-$d.png
done; done
bench/.venv/bin/python bench/grid_chart.py $t/epyc-higgs.png $t/dell-higgs.png \
    $t/epyc-mnist8m.png $t/dell-mnist8m.png --cols 2 --out $s/chart.png  # 2x2: HIGGS above MNIST8M
rm -r $t
```

The chess sets, unpruned, on the four nodes and dell2, each size after the
other on every node (`-a`: after the job that prepared them), and their charts:
one per size (`chart-chess_<size>.png`), and time and memory against the rows
(the runs of 2026-10-29/30: `submit-main-2026-10-29.sh`, below):

```bash
for d in chess_1m chess_10m chess_100m chess_1b; do
  bench/cluster/submit.sh -m 3 -d $d -p cart_depth10,c45_depth10_unpruned -a <prepare job> \
      cuda-hopper-$d cuda-ampere-$d cpu-epyc-$d cpu-dell-$d cpu-dell512-$d
  # one thread (--serial) on epyc1 and dell1; -a <build job> when it ran after a rebuild
  bench/cluster/submit.sh -m 3 -d $d -p cart_depth10,c45_depth10_unpruned serial-epyc-$d serial-dell-$d
  # --gpu-only on hopper and ampere (every CPU, as the CUDA runs)
  bench/cluster/submit.sh -m 3 -d $d -p cart_depth10,c45_depth10_unpruned gpuonly-hopper-$d gpuonly-ampere-$d
  # parallel on one socket of epyc1 and dell1
  bench/cluster/submit.sh -m 3 -d $d -p cart_depth10,c45_depth10_unpruned socket-epyc-$d socket-dell-$d
done
d=bench/results/cluster-chess
v=$d/dell-variance  # dell and dell512 at 1000M: the 10-run check (BENCHMARK.md)
bench/.venv/bin/python bench/testset_chart.py $d  # chart-chess_<size>.png: means ± sd, run counts
bench/.venv/bin/python bench/rows_chart.py $d/{cuda-hopper,cuda-ampere,cpu-epyc,cpu-dell,cpu-dell512,socket-epyc,socket-dell,serial-epyc,serial-dell,gpuonly-hopper,gpuonly-ampere}-chess_{1m,10m,100m,1b} \
    --out $d/chart-rows.png
bench/.venv/bin/python bench/runs_chart.py $d/cpu-dell-chess_1b $d/cpu-dell512-chess_1b --out $v/chart.png
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
- **`profile.sh`** queues the same job with `run.py --profile` on hopper,
  ampere and epyc1, then (one after the other on each node) again without it
  into `plain/`, to see what profiling costs:
  `cluster-ecbdl-profile` (removed in 254d4e8).
- **`scaling.sh`** queues the same job on epyc1 and dell1 with the thread list
  `1,2,4,8,12,…,48` (1 thread: `--serial`); `run.log` records the CPU order
  the threads take (`lscpu`).
- **`tune.sh`** runs a list of ./tree command lines on one node, interleaved
  and repeated, and prints ./tree's phase times with the CPU time, page faults
  and context switches of every run. The default settings were tuned
  with it ([docs/PERFORMANCE.md](../../docs/PERFORMANCE.md)).

## Test-set benchmarks

Since 2026-10-23 every dataset has a fixed training and test set, and these
benchmarks train ./tree on the training rows and score the tree on the test
rows (`submit.sh -e`: `run.py --test-accuracy`, the test rows predicted from
each `--dump`; the training accuracy is ./tree's own):

| Dataset | Training rows | Test rows |
|---|---|---|
| airline | 1987–2007 (109,258,555) | 2008, January to October (5,810,462; every row from the first with Year ≥ 2008; it has no published test set) |
| HIGGS | the first 10.5M | the last 500,000 (UCI's test set) |
| SUSY | the first 4.5M | the last 500,000 (UCI's test set) |
| ECBDL'14 | all 31,992,921 (`ecbdl`, CPU nodes only); the first 15M (`ecbdl_15m`, every node: the most an 80 GB GPU holds) | the competition's 2,897,917 |
| MNIST8M | infinite MNIST's 8,100,000 (LIBSVM `mnist8m`, 784 features, 10 classes) | the original MNIST test set, 10,000 (LIBSVM `mnist.t`) |
| SOREL-20M | its training set, 12,699,013 (2381 features; CPU nodes only, about 120 GB as float32) | its test set, 4,195,042 |

Two protocols, depth ≤ 10, no pruning: `cart_depth10` and
`c45_depth10_unpruned`. (Until 2026-10-27 also `cart_depth10_alpha`, α = 1e-5,
and `c45_depth10`, CF 0.25; those runs are in git at 96705a7.) Every dataset
runs CUDA hybrid (`cuda-*`) and GPU only (`gpuonly-*`) on hopper and ampere, and
on epyc and dell parallel on both sockets (`cpu-*`), parallel on one socket
(`socket-*`), `--serial` (`serial-*`) and `--simple` (`simple-*`). Not run: the
GPUs on SOREL and full ECBDL (they do not fit in 80 GB of VRAM), and
`socket-epyc-sorel` (one epyc socket has 256 GB, SOREL needs 340–370 GiB).
Runs per case: 10 when a run (the whole process) takes about 2 minutes or less
(chess 30), 3 under about 10 minutes, longer ones once; a bar under 10 minutes
of training always has at least 3. `testset_chart.py` groups the bars
by backend, the nodes under each, and draws the means on a log scale with
whiskers from the fastest to the slowest run (training time ± standard
deviation, the number of runs, peak RAM and VRAM as numbers, the tree's
accuracy, nodes and depth); `--table` prints the same as Markdown. It marks
potential problems from `cluster-<dataset>/problems.json` and runs over twice
the mean of the other runs (none now).

### Reruns of 2026-10-29/30

Every run that main's CPU presort (4fb419c) made out of date, every backend, on
one build of main: `git archive ec77727` into `~/dt-main`, built by job 880 with
`-falign-functions=64 -falign-loops=32` and installed as `~/dt/tree_main` (CUDA
and CPU, `TREE_CUDA_BIN`) and `~/dt/tree_cpu_main` (`TREE_BIN`).
`submit-main-2026-10-29.sh` made the plans and queued jobs 881–886 (epyc1, epyc2,
dell1, dell2, hopper, ampere). Kept: `--simple` runs not rerun (its code has not
changed), the libraries' runs over 10 minutes, and the GPU libraries of
cluster-gpu-vs-others. Chess moved to `cart_depth10`, `c45_depth10_unpruned`
with 30 runs a case.

Top-ups of 2026-10-30, one plan line each, by hand: job 972, two more runs of
`serial-epyc-ecbdl` CART, merged with `merge_runs.py`; jobs 973 and 974, dell1's
4-thread repeat and old-against-new runs, in
`cluster-cpu-scaling/dell-ab/` ([its BENCHMARK.md](../results/cluster-cpu-scaling/BENCHMARK.md#dell1-two-oddities-checked-on-2026-10-30)).

### Reruns of 2026-10-28/29

Replaced by the runs above; every run folder of the test-set benchmarks and
cluster-chess was run on one build per backend:

| Backend | Binary on mordor3 | Source |
|---|---|---|
| CUDA hybrid, GPU only | `~/dt/tree` | 67495f9, the CUDA rework (installed by job 636 after its trees matched the old build's) |
| parallel, both sockets and one | `~/dt/tree_cpu3` (`TREE_BIN`; git describe `67495f9-dirty-poll`) | the CPU code committed in 8bdf641: the waits below, built with `-falign-functions=64 -falign-loops=32` |
| `--serial`, `--simple` | `~/dt/tree_cpu` | the CPU build of 2026-10-28: the same source except the parallel waits, which these backends do not use (and without the alignment flags) |
| `--simple` of the night of 2026-10-27/28 (epyc: airline, HIGGS, SUSY, ECBDL; MNIST8M and SOREL on both nodes) | `~/dt/tree_cpu` as it was then | 8187eb9: --simple's code is the same as in the build above, so those runs were kept |

Every tree is identical to the old builds' (E and G dumps: 20 pairs, ECBDL 15M
and chess_10m). The plans and logs are in `plans/`; `submit-batches-2026-10-28.sh`
(jobs 636–645) and `submit-poll-2026-10-29.sh` (jobs 771 on) made them: `submit.sh -P
<plan>` writes a run's line to a plan file instead of submitting it, and
`batch.sh <plan>` (one SLURM job per node and group) runs the lines one after
another, each into its folder's `run.log`, going on when one fails. Long runs
went on epyc2 and dell1, so epyc1 stayed free for short and interactive runs.
The other `submit-*-2026-10-2{8,9}.sh` are the steps before, each with an
as-run header; their runs were all replaced.

Why the parallel backend was rebuilt three times:

- **Spin stall (epyc).** `partitionInBlocks` (`src/build/cpu_builder.cpp`)
  splits a big node in blocks taken in order by all threads, each waiting for
  the block before by spinning on `_mm_pause` without end. When that block's
  thread was descheduled the spinners kept it off a CPU: on epyc chess_10m,
  13 of 50 runs took over twice the median.
- **Spin fix with `atomic::wait`.** 2,048 pauses, then `atomic::wait`. That
  made dell's chess about 2× slower in some node states (dell2 chess_100m
  2.8 s instead of 1.25 s), and libstdc++ 13 can lose its wake-up: `notify`
  skips the futex wake when it sees no waiter, and that check can pass the
  release store, so about 1 in 40 big dell runs hung (job 747).
- **Final wait (`waitNonZero`).** 2,048 pauses, then `yield` for up to 2 ms,
  then 50 µs sleeps; release stores and no notify, so no wake-up can be lost.
- **Code alignment.** An intermediate build was 8% slower on epyc's ECBDL 15M
  C4.5 (7.80 → 8.41 s). The cause was not the waits or seq_cst but C4.5's sweep
  (`TwoClassSweep::advance`) landing on another offset mod 64;
  `-falign-functions=64 -falign-loops=32` restores it (A/B jobs 763, 764, 766).

The final build against the old one (parallel): on epyc ECBDL C4.5 1.02–1.03×
the old time (15M 7.82 → 7.99 s, full 17.7 → 18.1 s), CART 0.99×, MNIST8M
C4.5 0.95×, chess_100m 0.93–0.95×; on dell chess 100M and 1B about twice as
fast (chess_1b CART 23.6 → 10.8 s), elsewhere up to 14% faster (SUSY CART) or
within 3%. The CUDA rework is 10–25% faster than the old CUDA build on HIGGS,
airline and ECBDL 15M.

The reruns wrote into `cluster-<dataset>-fix/` and `-par2/`; once done, each
of their run folders replaced the one of the same name (no merging):

```bash
cd bench/results
for x in airline higgs susy ecbdl mnist8m sorel chess; do
  for f in fix par2; do
    for run in cluster-$x-$f/*/; do rm -rf cluster-$x/$(basename $run); mv $run cluster-$x/; done
    rm -rf cluster-$x-$f
  done
done
```

The first submits (2026-10-23 to 27), for a new dataset or node:

```bash
P=cart_depth10,c45_depth10_unpruned
for d in higgs airline susy ecbdl_15m; do   # -a <prepare job> when the data is still being made
  bench/cluster/submit.sh -m 3 -e -d $d -p $P {cuda,gpuonly}-{hopper,ampere}-$d cpu-{epyc,dell}-$d serial-epyc-$d
done
bench/cluster/submit.sh -m 3 -e -d ecbdl -p $P cpu-epyc-ecbdl cpu-dell-ecbdl serial-epyc-ecbdl
bench/cluster/submit.sh -m 3 -e -d mnist8m -p $P {cuda,gpuonly}-{hopper,ampere}-mnist8m \
  {cpu,socket,serial}-dell-mnist8m
bench/cluster/submit.sh -m 3 -e -d mnist8m -p $P -w epyc1 {cpu,socket,serial}-epyc-mnist8m
bench/cluster/submit.sh -m 3 -e -d sorel -p $P cpu-dell-sorel socket-dell-sorel
bench/cluster/submit.sh -m 1 -e -d sorel -p $P serial-dell-sorel
bench/cluster/submit.sh -m 3 -e -d sorel -p $P -w epyc2 cpu-epyc-sorel
bench/cluster/submit.sh -m 1 -e -d sorel -p $P -w epyc2 serial-epyc-sorel
# --simple: one job per protocol and one run, straight to job.sh (2026-10-27:
# bench/cluster/submit-mnist8m-sorel.sh on mordor3); e.g. SOREL on dell1, 3 days:
n=simple-dell-sorel; mkdir -p bench/results/cluster-sorel/$n
for p in cart_depth10 c45_depth10_unpruned; do
  sbatch -J dt-$n-${p%%_*} -p dell-long -w dell1 --exclusive -t 3-00:00:00 \
    --export=ALL,ACCURACY=--test-accuracy -o bench/results/cluster-sorel/$n/run-${p%%_*}.log \
    bench/cluster/job.sh cluster-sorel $n sorel tree_simple 1 $p 1 172800
done
for d in airline higgs susy ecbdl mnist8m sorel; do bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-$d; done
```

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
- **The datasets leave the page cache before every run** (`run.py
  --drop-cache`, from `job.sh`). The cached CSVs of earlier jobs pile up on
  one NUMA node: on 2026-10-20 they held 236 of the 252 GiB of epyc1's node 1,
  so a 48-thread run's pages went to node 0 and half the threads read across
  the sockets (ECBDL presort 1.6 → 2.3 s). Each run records its free memory
  per node in `results.jsonl` (`mem_free_node*_kb`).
- **Jobs leave nothing on the node.** The datasets live only in
  `bench/data` on the login node (the other checkouts symlink it). `job.sh`
  and `build.sh` point `TMPDIR` at a folder of the job (`/tmp/dt-<job>.*`;
  /tmp is tmpfs, so leftovers hold RAM) and at exit, also a failed or
  timed-out one, `cleanup.sh` removes it and drops the datasets from the
  node's page cache.
- **RAM in the chart** is the kernel's MemTotal with the module type and
  speed. udev's DMI list, the usual source, is cut short on these boards
  (11–15 of 16 modules).

## Data

On mordor3 every dataset lives once, in `~/datasets`, outside the clones, and
that folder holds only the CSVs (`~/dt/datasets` is a link to it). Everything
derived or downloaded is in `~/dt/bench/data` (other clones link to it):

```
~/datasets/
  higgs.csv  supersymmetry.csv  airline.csv  covertype.csv
  ecbdl/   ecbdl.csv (training set), ecbdl_test.csv (the competition's test set)
  chess/   chess_{1m,10m,100m,1b,2b}.csv, chess_{1m,10m,100m,1b}_test.csv
~/dt/bench/data/
  <name>/                 bench/prepare.py's output, one folder per datasets.toml entry
  downloads/              the downloads make_datasets.py reads (ecbdl: a link to /home/Datasets/ECBDL)
  supersymmetry_20k.csv   the golden tests' sample (tests/check.sh)
```

A prepared folder holds the split in each tool's format. A chess set's
`train.csv` is a link to its CSV in `~/datasets/chess/`; `chess_50m` and
`chess_500m` take only the first rows of a bigger set, so theirs are files.

HIGGS and SUSY are in UCI's row order, so their last 500,000 rows are the
published test set: `datasets.toml` takes them with `split = "tail"`, and
./tree reads the CSV directly with `--test-last 500000`.

The cluster gets its data from the internet, not from the desktop (whose
upload is slow). `make_datasets.py` (run by `prepare.sh`) rebuilds
`covertype.csv` and `higgs.csv` byte for byte as the desktop copies, makes the
others, and stops on any SHA-256 difference:

| File | Source | SHA-256 (= desktop; airline: first build) |
|---|---|---|
| `covertype.csv` | UCI 31 `covertype.zip` (same rows and order; `Id`, header and class names added) | `fd7a387d…` |
| `higgs.csv` | OpenML 45570 parquet: UCI HIGGS, same rows and order. UCI's own zip downloads at 0.5 MB/s, OpenML's at 19 MB/s. The parquet's doubles are slightly off the float32 values, so they are rounded back to float32 and printed as UCI does (`%.18e`, CRLF line ends, `Id`, label last as signal/background) | `71f566a2…` |
| `supersymmetry.csv` | UCI 279 `susy.zip`: UCI's rows, order and text (since 2026-10-23; `Id` first, the label last as signal/background), so the last 500,000 rows are the published test set. Until then it was the desktop copy: the same float32 values and labels, but its rows shuffled (no common seed reproduces the order) and its doubles UCI's text read by an unknown parser, 0–6 ulps off (whole file `85ecf105…`). The benchmarks on SUSY before 2026-10-23 used that order | `d152c759…` |
| `supersymmetry_20k.csv` (in `bench/data/`) | The desktop copy's first 20,000 rows, the sample of the golden trees in `tests/golden/` (`tests/check.sh`). Two files made from the desktop copy carry what is not public: `susy_order.u64` (40 MB, a hash of every row in the desktop's order) and `susy_delta.i8` (90 MB, every value's offset in float64 ulps; 21 MB zstd) | `4563a738…` |
| `airline.csv` | gbm-bench's airline dataset (also in the XGBoost GPU paper): US flights from October 1987 to April 2008, `airline_14col.data.bz2` from kt.ijs.si (1.09 GB, SHA-256 `1f13460f…`). 115,069,017 rows, 13 integer features; the carrier and the airports become codes in sorted order, as in gbm-bench, and the label is `ArrDelay > 0` (`delayed` 47.5%, `on_time` 52.5%). Rows in file (date) order, so `datasets.toml` takes a random 500,000-row test set. There is no desktop copy: the hash is that of its first build. It takes about 3 minutes, and `prepare.py` 5 more | `25542603…` |

Airline has 10.5× HIGGS's rows and 4.9× its values (1.49 G against 0.31 G;
6.0 GB as float32 against 1.2 GB). Its CSV is smaller than HIGGS's only
because its values are short integers. It is cluster-only: the CUDA backend
needs 13.3 GiB of GPU memory for it, more than the laptop's 8 GB or a 2080 Ti's 11 GB.
`ActualElapsedTime` partly gives the delay away, so its accuracy says little;
it is there to compare speed, as in gbm-bench.

`ecbdl.csv` is ECBDL'14 (protein contact-map prediction, the GECCO 2014 big
data competition) from the ARFF in mordor3's `/home/Datasets/ECBDL` (60.8 GB):
31,992,921 rows, 631 features, label 0/1 (about 2% contacts). The 539 real
attributes become float32; the 92 nominal ones (predicted secondary structure
H/E/C, solvent accessibility 0–4, X where the window runs past the chain) become
integer codes in sorted order, as airline's airports. Its 20.2 G values are 65×
HIGGS's (81 GB as float32). Since 2026-10-23 `datasets.toml`'s `ecbdl` is the
competition's split: every training row (`train.csv` is a link to `ecbdl.csv`,
which keeps its own feature names and labels 0/1; `prepare.py` records them in
`meta.json` as `csv_features`/`csv_classes`) and `ecbdl_test.csv` as the test
set. Before that it set 100,000 random rows aside as the test set and trained on
the other 31,892,921 (the runs in `cluster-ecbdl-old`, removed; last in f4668df); it writes
no YaDT files. The CUDA backend needs 8 bytes per value
(one copy of the presorted columns, plus one spare column), about 160 GB for the
full set, so the GPUs get `ecbdl_15m`: a random 15M of the training rows (9.5 G
values, about 72 GiB of GPU memory). Until 2026-10-17 the GPU needed 16 bytes per
value and the sample had 6M rows. `ecbdl.csv`: 59,533,971,938 bytes, SHA-256
`5bc39aec…` (first build). Logs: `prepare-ecbdl.log` (the full set; it also made the earlier 6M sample), `prepare-ecbdl_15m.log` (the 15M sample, `bench/prepare.py ecbdl_15m` alone in a job on dell3). The pre-converted `.bin` files under
`/home/Datasets/IMM` lack the label and the nominal attributes.

`ecbdl/ecbdl_test.csv` is the competition's own test set, with the labels
published after it closed (2014-07-07): `test_set.arff.xz` from
<http://cruncher.ico2s.org/bdcomp/> (121 MB, SHA-256 `ac26c4cf…`; not kept,
download it again to rebuild). It has 2,897,917 rows
(label 1: 48,637, 1.68%) and the same ARFF header as the training set.
`bench/cluster/ecbdl_test.py` converts it with `make_datasets.py`'s encoding,
taking the nominal codes from the training ARFF's header (checked: it turns the
training ARFF's first 1,000 rows into `ecbdl.csv`'s exact bytes). The result has
5,423,008,790 bytes, SHA-256 `3d838532…`, and the same header as `ecbdl.csv`. To rebuild it:
`xz -dk test_set.arff.xz`, then `bench/.venv/bin/python -I
bench/cluster/ecbdl_test.py /home/Datasets/ECBDL/training_set.arff
test_set.arff datasets/ecbdl/ecbdl_test.csv` on a compute node. It is `ecbdl`'s test set (see above);
`ecbdl_15m` keeps its 100,000 random training rows.

The **chess** sets are a 3×3 checkerboard: x and y uniform on a grid of 32,768
values in [0, 3] (3k/32767 for k = 0..32767 to six decimals, a C `rand()` with
`RAND_MAX` 32767; both 0 and 3 occur, no inner square boundary does), the class
alternating over the nine unit squares (`B` on 5 of them, 55.5% of the rows), no
noise; a tree of 9 leaves and depth 4 is exact. They came as C4.5 files
(`.names`, `.data`, `.test`) in mordor3's `~/chess`, not public:
`chess3x3x001M`, `010M`, `100M` and `001MM` (1M, 10M, 100M and 1000M rows).
`make_datasets.py` turned them into `datasets/chess/chess_*.csv` and `chess_*_test.csv`
(the class without C4.5's closing period). The sources were uneven: 10M and up
had CRLF line ends, 100M ended with DOS's Ctrl-Z (byte 0x1a, which ./tree cannot
parse), and their shared test file spaced its fields (`0.182, 0.025, B.`). The
sizes nest: 1M is the first million rows of 10M, 10M the first 10M of 100M, and
**1000M is the 100M file ten times over** (with its Ctrl-Z after each copy), so
it has only 100M distinct rows. 1M and 1000M share one test file (1,000 rows,
the generator's), 10M and 100M another (4,500 rows on a grid of 3 decimals).
The originals were deleted after the conversion (2026-10-19): the CSVs are the
only copies, and `chess-originals.sha256` lists the originals' SHA-256.

Since 2026-10-22 the CSVs are in ./tree's format, the bytes `prepare.py` writes
as `train.csv` (header `"f0","f1","class"`, classes `c0` = A and `c1` = B,
values as float32's shortest text), so `prepare.py` links a set's `train.csv` to
its CSV when the set takes all of the CSV's rows, instead of a second copy
(`chess_tree_format.py` converted the first build and replaced the existing
`train.csv` files with links; the sets prepared before keep `x`, `y`, `A`, `B` in
`meta.json`). `make_datasets.py` also **generates chess_2b** (2,000,000,000
rows; ./tree's limit for two classes is 2^31): the 100M rows of `chess_100m`,
then 1.9G rows drawn the same way (k uniform on the grid, numpy's
`default_rng([20261022, chunk])` per 10M rows), after checking that every row of
`chess_100m` is on the grid and follows the squares' rule. `datasets.toml` adds
`chess_50m` (the first 50M rows of `chess_100m`) and `chess_500m` (the first
500M of `chess_2b`), with `chess_100m`'s test file, as `chess_2b` has. Each set
takes its own test file (`test_source`), and none writes YaDT files from 50M on.
CSVs: about 20.8 bytes per row, 20.8 GB for 1000M, 41.6 GB for 2b (generated in 164 s on dell1). Logs:
`prepare-chess.log` (the first four, 5 minutes), `prepare-chess-2b.log` (the
conversion, 3 minutes; chess_2b; preparing 50m, 500m and 2b, 5 minutes and 170 GB of RAM).

HIGGS's prepared files (`bench/data/higgs/`) have the SHA-256 recorded in
`plan.json` of `consumer/pc-legion-higgs/cpu-pc-higgs` and `consumer/pc-legion-higgs/cuda-legion-higgs` (desktop and laptop).
Logs: `prepare-higgs.log`, `prepare-airline.log`.
