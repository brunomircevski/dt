# ./tree on the mordor3 cluster

Scripts that build, prepare and benchmark ./tree on the faculty's mordor3
cluster (Politechnika Białostocka, SLURM), and tune it there. The results and
what each benchmark shows: [bench/results/](../results/README.md).

**The GPU nodes (hopper, ampere) run CUDA only, the CPU nodes (epyc1, dell1)
parallel only:** a GPU node's CPU is there to feed the GPU and is not
benchmarked on its own. epyc2 is the same as epyc1 (within 2%) and takes the
long one-thread runs; dell2–4 have half dell1's memory channels and 512 GB, up
to 1.46× slower, so they are not used. turing (RTX 2080 Ti) is not used.

| Node | CPU | RAM | GPU (GPU used: hopper 2, ampere 0) |
|---|---|---|---|
| hopper | 2× Xeon Gold 6444Y (Sapphire Rapids), 32 cores | 1 TB DDR5-4800 | 4× H100 PCIe 80 GB |
| ampere | 2× EPYC 7402 (Zen 2), 48 cores | 1 TB DDR4-3200 | 2× A100 80 GB + 2× A100 40 GB PCIe (GPU 0, used: 80 GB) |
| epyc1 | 2× EPYC 9255 (Zen 5), 48 cores | 512 GB DDR5-6400 | – |
| dell1 | 2× Xeon Gold 6542Y (Emerald Rapids), 48 cores | 1 TB DDR5-5200 | – |

## Run it

On mordor3, in a clone of this repository. Nothing heavy runs on the login
node: every step is a SLURM job. SLURM's commands are in `/opt/slurm-25.05.8/bin`
(`/etc/profile.d/slurm.sh`), and the partitions are hidden (`sinfo -a`).

```bash
# once: the Python venv (python3-venv is missing there; virtualenv is installed)
virtualenv -p python3 bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt
# once: the downloads, see "Data" below, into bench/data/downloads
sbatch --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)" \
    bench/cluster/build.sh       # on hopper: ./tree for sm_75 + sm_80 + sm_90, ./tree_cpu, rusage
sbatch bench/cluster/prepare.sh  # on a dell node: the datasets (checked by SHA-256), HIGGS prepared
```

### The final rerun

`final.sh` queues every run of the benchmarks in bench/results (except
cluster-profiling, hybrid-vs-gpu-only and the power benchmark) on one build:
one chain of `batch.sh` jobs per node, each running a plan file
(`bench/cluster/plans/final-<node>-<n>.txt`, its log next to it) line by line,
the most important first. Started on 2026-11-09 from `~/dt-final`, a clone
with its own build whose `bench/data`, `bench/.venv` and `datasets` link to
`~/dt`'s (and a copy of `bench/.tools`), so rebuilding `~/dt` cannot change it.

```bash
git clone https://github.com/brunomircevski/dt ~/dt-final && cd ~/dt-final
ln -s ~/dt/bench/data bench/data; ln -s ~/dt/bench/.venv bench/.venv; ln -s ~/datasets datasets
cp -a ~/dt/bench/.tools bench/.tools
sbatch --export=ALL,TREE_GIT_DESCRIBE="$(git describe --always --dirty --abbrev=12)" bench/cluster/build.sh
bench/cluster/final.sh -t   # the check: one run of each kind of job, into bench/results/cluster-test/
bench/cluster/final.sh -n   # the plans only; then move the old run folders they name out of bench/results
bench/cluster/final.sh      # submit
```

| Node | Jobs, in order | Runs | Expected |
|---|---|---|---|
| hopper, ampere | `--cuda` and `--gpu-only` on SUSY, HIGGS, MNIST8M, ECBDL'14 15M, SOREL 4M, chess 1M–1000M; then cuML and XGBoost (one tree, then the default forest of 10) on HIGGS and MNIST8M from `~/gpu-dt-bench` into `results-final/` | 10 | half a day each |
| epyc1 | `--parallel` 48 threads on every dataset, thread scaling (HIGGS, MNIST8M), `--serial` on every dataset, `--simple` on all but ECBDL'14, SOREL and SOREL 4M | 10 | about 2.7 days |
| epyc2 | scikit-learn and Weka J48 on HIGGS and MNIST8M (cluster-cpu-vs-others), then `--simple` on ECBDL'14, SOREL and SOREL 4M | 10; J48 on MNIST8M and `--simple` on full ECBDL'14 and SOREL 3 | about 3 days |
| dell1 (dell-long) | the same as epyc1, then the long `--simple` runs | as epyc | about 5.5 days |

Runs per case: 10, and 3 where one process takes over an hour. Parallel runs
use every CPU (both sockets), one-thread runs the first CPU, memory
first-touch; CUDA runs one GPU and every CPU, the GPU's socket first. Every
run records its test accuracy and the SHA-256 of its tree. The datasets the
GPUs cannot hold (full ECBDL'14 and SOREL) run on the CPU nodes only.

When a chain is done, copy its run folders into this repository's
bench/results (scp) and redraw the charts on the PC:

```bash
for d in susy higgs mnist8m ecbdl sorel chess; do bench/.venv/bin/python bench/testset_chart.py bench/results/cluster-$d; done
bench/.venv/bin/python bench/scaling_chart.py bench/results/cluster-cpu-scaling
bench/.venv/bin/python bench/vs_chart.py cpu
bench/.venv/bin/python bench/vs_chart.py gpu   # after copying ~/gpu-dt-bench/results-final/<data>/<node>/
bench/.venv/bin/python bench/rows_chart.py bench/results/cluster-chess/{cuda-hopper,cuda-ampere,cpu-epyc,cpu-dell,serial-epyc,serial-dell,gpuonly-hopper,gpuonly-ampere}-chess_{1m,10m,100m,1b} \
    --out bench/results/cluster-chess/chart-rows.png
```

- **`build.sh`** runs on hopper because the CPU nodes have no CUDA toolkit. It
  builds with `NVCC=/usr/local/cuda/bin/nvcc` and `-gencode` for sm_75, sm_80
  and sm_90, so one binary serves the A100, the H100 and turing's RTX 2080 Ti, and
  rebuilds the CUDA object when that list changes. Log: `build.log` (not in git).
- **`final.sh`** queues every final run, above.
- **`submit.sh`** queues one job per node (`job.sh`) with `--exclusive` and
  `--hold`, then releases them together, so on idle nodes they start in the
  same second. Each job runs
  `bench/run.py <dataset> --protocols <-p> --impls tree|tree_cuda --threads all --cpus 0-<cores-1> --reps <-m> --run-id cluster-<…>/<run> --train-accuracy-only`
  and `report.py`: `tree_cuda` on the GPU nodes, `tree` (parallel) on the
  CPU nodes. Its log is `<run>/run.log`. With `-P <plan>` it writes a run's line
  to a plan file instead, and `batch.sh <plan>` (one SLURM job per node) runs the
  lines one after another, going on when one fails: that is how long reruns go.
- **`profile.sh`** queues the same job with `run.py --profile` on hopper,
  ampere and epyc1, then (one after the other on each node) again without it
  into `plain/`, to see what profiling costs.
- **`scaling.sh`** queues the same job on epyc1 and dell1 (by hand; final.sh does it too) with the thread list
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
| HIGGS | the first 10.5M | the last 500,000 (UCI's test set) |
| SUSY | the first 4.5M | the last 500,000 (UCI's test set) |
| ECBDL'14 | all 31,992,921 (`ecbdl`, CPU nodes only); the first 15M (`ecbdl_15m`, every node: the most an 80 GB GPU holds) | the competition's 2,897,917 |
| MNIST8M | infinite MNIST's 8,100,000 (LIBSVM `mnist8m`, 784 features, 10 classes) | the original MNIST test set, 10,000 (LIBSVM `mnist.t`) |
| SOREL-20M | its training set, 12,699,013 (`sorel`, 2381 features; CPU nodes only, about 120 GB as float32); the first 4M (`sorel_4m`, every node: the most an 80 GB GPU holds) | its test set, 4,195,042 |

Two protocols, depth ≤ 10, no pruning: `cart_depth10` and
`c45_depth10_unpruned`. (Until 2026-10-27 also `cart_depth10_alpha`, α = 1e-5,
and `c45_depth10`, CF 0.25; those runs are in git at 96705a7.) Every dataset
runs CUDA hybrid (`cuda-*`) and GPU only (`gpuonly-*`) on hopper and ampere, and
on epyc and dell `--parallel` with 48 threads on both sockets (`cpu-*`),
`--serial` (`serial-*`) and `--simple` (`simple-*`); the chess sets too. Not
run: the GPUs on full SOREL and full ECBDL (they do not fit in 80 GB of VRAM;
the GPUs get `sorel_4m` and `ecbdl_15m`). Runs per case: 10, and 3 where one
process takes over an hour (final.sh). Until 2026-11-09 there were also
one-socket runs (`socket-*`), dell2 (`cpu-dell512-*`, chess) and airline; they
are in git at 3354a8a. `testset_chart.py` groups the bars
by backend, the nodes under each, and draws the means on a log scale with
whiskers from the fastest to the slowest run (training time ± standard
deviation, the number of runs, peak RAM and VRAM as numbers, the tree's
accuracy, nodes and depth); `--table` prints the same as Markdown. It marks
potential problems from `cluster-<dataset>/problems.json` and runs over twice
the mean of the other runs (none now).

### Which build ran

Every run of the final rerun (2026-11-09 on) is one build of main in
`~/dt-final` (final.sh); each run's `machine.json` names its commit. Until it
replaces them, the folders hold the runs of 2026-10-29/30 on ec77727 (`--simple`
from 2026-10-27/29, SOREL's `--simple` on 32cf657, `sorel_4m` on 4ece22f), with
the CUDA runs on the GPU's socket only.

## Settings specific to the cluster

The data and measurements are those of the other benchmarks
([bench/README.md](../README.md)), with every CPU of each node:

| Backend | Command | Threads |
|---|---|---|
| Parallel (epyc1, dell1) | `./tree_cpu --parallel --threads N` | N = every CPU (48) |
| CUDA (hopper, ampere) | `./tree --cuda --threads N` | one GPU, plus N CPU threads on both sockets (hopper 32, ampere 48) for the small subtrees and pruning |

- **Whole nodes.** SLURM on mordor3 hands out cores only: it tracks no GPUs
  (no GRES) and no memory. Every job therefore takes its node with
  `--exclusive`, so no other job shares its cores, memory bandwidth or GPUs.
  `run.log` records the load average and the GPU processes at the start.
- **One GPU, every CPU.** job.sh takes the NUMA node with a GPU and the most
  free memory and the first GPU on it (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES`): hopper GPU 2 on node 1 (another user's /dev/shm
  fills node 0), ampere GPU 0 on node 0, an A100 80 GB. The threads take that
  node's CPUs first, then the other socket's (hopper 16–31, 0–15; ampere
  0–23, 24–47), and the memory is not bound: binding it to the GPU's node was
  slower in every deep case. On ampere, CUDA hybrid at no depth limit was
  1.2–1.5× faster on both sockets than on the GPU's (HIGGS, SUSY, airline
  CART), and within ±6% at depth 10 (2026-11-06). `run.py` samples that
  GPU's memory (`gpu_index` in `machine.json`). The other GPUs sit idle.
  The depth-10 runs of ec77727 predate this and ran on the GPU's socket
  only (`machine.json` and `run.log` record the CPUs); the final rerun does not.
- **CPU governors are as the admins set them.** epyc1 `performance`, dell1
  `powersave` (intel_pstate, which still boosts), hopper
  `schedutil`, ampere `ondemand`. A user can't change them; `machine.json`
  records them. All four dell nodes are set alike (checked on 2026-10-14:
  `powersave` on all 48 CPUs, intel_pstate, energy-performance preference
  `balance_performance`, turbo on), so dell1 stands for all of them. With
  intel_pstate, `powersave` is the dynamic governor that still reaches turbo,
  not acpi-cpufreq's `powersave`, which holds the lowest frequency.
- **dell is noisy at 2–12 threads** (CV 13–17% at 4 threads, cause unknown), and
  its batches differ by up to 35%; epyc is steady
  ([docs/PERFORMANCE.md](../../docs/PERFORMANCE.md)). Compare low thread counts
  on epyc, and on dell only interleaved in one job.
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
set. It writes no YaDT files. The CUDA backend needs 8 bytes per value
(one copy of the presorted columns, plus one spare column), about 160 GB for the
full set, so the GPUs get `ecbdl_15m`: a random 15M of the training rows (9.5 G
values, about 72 GiB of GPU memory). `ecbdl.csv`: 59,533,971,938 bytes, SHA-256
`5bc39aec…` (first build). The pre-converted `.bin` files under
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
only copies.

Since 2026-10-22 the CSVs are in ./tree's format, the bytes `prepare.py` writes
as `train.csv` (header `"f0","f1","class"`, classes `c0` = A and `c1` = B,
values as float32's shortest text), so `prepare.py` links a set's `train.csv` to
its CSV when the set takes all of the CSV's rows, instead of a second copy (the
sets prepared before keep `x`, `y`, `A`, `B` in `meta.json`). `make_datasets.py` also **generates chess_2b** (2,000,000,000
rows; ./tree's limit for two classes is 2^31): the 100M rows of `chess_100m`,
then 1.9G rows drawn the same way (k uniform on the grid, numpy's
`default_rng([20261022, chunk])` per 10M rows), after checking that every row of
`chess_100m` is on the grid and follows the squares' rule. `datasets.toml` adds
`chess_50m` (the first 50M rows of `chess_100m`) and `chess_500m` (the first
500M of `chess_2b`), with `chess_100m`'s test file, as `chess_2b` has. Each set
takes its own test file (`test_source`), and none writes YaDT files from 50M on.
CSVs: about 20.8 bytes per row, 20.8 GB for 1000M, 41.6 GB for 2b (generated in 164 s on dell1).
