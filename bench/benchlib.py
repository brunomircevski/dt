"""Shared paths, dataset files and protocol definitions for the CPU benchmark.

Data layout (bench/data/<dataset>/, made by prepare.py; on mordor3 bench/data
and the repository's datasets/ are both links to ~/datasets): one folder per
dataset, with its prepared files below and its originals (the CSVs
datasets.toml reads, the downloads they are made from) side by side:

  meta.json                  features, classes, row counts, how the split was made
  train.f32, test.f32        features, row-major little-endian float32 (n x F)
  train.y.i32, test.y.i32    class index per row, little-endian int32
  train.csv                  header f0..f{F-1},class; labels c0..c{K-1}, or
                             a link to a source CSV with names of its own (ECBDL):
                             meta.json's csv_features / csv_classes (for users;
                             the benchmarks give ./tree train.f32, the same values)
  yadt.names, train.yadt.csv, test.yadt.csv   headerless CSV + metadata  (YaDT)
  <name>.meta.json           the meta.json of a prefix_of set <name> of this
                             dataset (ecbdl_15m in ecbdl/, sorel_4m in sorel/):
                             no folder of its own

Several datasets may share one folder (datasets.toml folder = "chess": chess_2b,
chess_1b and chess_2b's prefixes): each is then named by its own stem,
<name>.meta.json, <name>.f32, <name>.y.i32, <name>.csv (training rows only:
no test set, no YaDT files). stem() gives a dataset's files for either layout.

A dataset without a test set (split = "none": the synthetic chess sets) has
n_test 0 and no test files. A dataset that is another one's first rows
(prefix_of in datasets.toml: sorel_4m, ecbdl_15m, chess_100m) has only its
<name>.meta.json, in its parent's folder (meta_path()): every tool reads its
parent's files (folder()), ./tree its train.f32 with --rows (tree_input()), the
others the binary files' first n_train rows.

Every tool sees the same float32 values: the CSVs print them exactly (9
significant digits), the binary files hold them as they are.
"""

import functools
import json
import os
import subprocess
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
DATA = BENCH / "data"
TOOLS = BENCH / ".tools"
RESULTS = BENCH / "results"
TREE = Path(os.environ.get("TREE_BIN", ROOT / "tree_cpu"))
TREE_CUDA = Path(os.environ.get("TREE_CUDA_BIN", ROOT / "tree"))  # `make`: CPU + CUDA
YADT = TOOLS / "yadt" / "dTcmd"
RUSAGE = TOOLS / "rusage"  # adapters/rusage.c: runs every measured process
JAVA_CLASSPATH = f"{TOOLS / 'weka.jar'}:{TOOLS / 'bounce.jar'}:{TOOLS / 'classes'}"
BASELINE = "_baseline"  # tiny synthetic dataset: each tool's runtime footprint
# The files prepare.py writes in a dataset's folder (the rest are originals).
PREPARED_FILES = ("meta.json", "train.f32", "train.y.i32", "test.f32", "test.y.i32", "train.csv",
                  "yadt.names", "train.yadt.csv", "test.yadt.csv")

# CART trees are capped at depth 30 for every tool: rpart cannot grow deeper,
# and a tree that only one tool may grow deeper is not the same work.
CART_DEPTH = 30

# The fixed alpha of cart_alpha (run.py --alpha overrides it; ./tree gets it as --prune). Every CART tool
# gets this same number; nothing tunes it per tool or per dataset, and no
# tool's tree is forced to another tool's size: each reports the tree it builds.
# No cross-validation anywhere in the benchmark: every case grows one tree.
CART_ALPHA = 1e-5

# ------------------------------------------------------------------ protocols
#
# A protocol is one well-defined piece of work that every listed implementation
# performs with equivalent settings. "{alpha}" is filled in by run.py
# (CART_ALPHA or --alpha).
#
# ./tree and YaDT entries are command-line flags; the others are key=value
# arguments of the adapters in bench/adapters/.

PROTOCOLS = {
    "cart_full": {
        "title": "CART, grown to pure leaves (depth <= 30), no pruning",
        "impls": {
            "tree": ["--cart", "-d", str(CART_DEPTH)],
            "sklearn": {"mode": "fit", "max_depth": CART_DEPTH},
            "rpart": {"mode": "fit", "maxdepth": CART_DEPTH},
        },
    },
    "cart_depth12": {
        "title": "CART, grown to depth 12, no pruning",
        "impls": {
            "tree": ["--cart", "-d", "12"],
            "sklearn": {"mode": "fit", "max_depth": 12},
            "rpart": {"mode": "fit", "maxdepth": 12},
        },
    },
    "cart_alpha": {
        "title": "CART, one tree pruned at a fixed alpha (cost-complexity)",
        "impls": {
            "tree": ["--cart", "-d", str(CART_DEPTH), "--prune", "{alpha}"],
            "sklearn": {"mode": "fit", "max_depth": CART_DEPTH, "ccp_alpha": "{alpha}"},
            "rpart": {"mode": "alpha", "maxdepth": CART_DEPTH, "alpha": "{alpha}"},
        },
    },
    # cart_alpha and cart_full without the depth cap, for ./tree-only
    # benchmarks (thread scaling, the mordor3 cluster): the cap exists only
    # because rpart cannot grow deeper than 30, and C4.5 has none.
    "cart_alpha_nodepth": {
        "title": "CART, one tree pruned at a fixed alpha (cost-complexity), no depth limit",
        "impls": {
            "tree": ["--cart", "--prune", "{alpha}"],
            "sklearn": {"mode": "fit", "ccp_alpha": "{alpha}"},
        },
    },
    "cart_full_nodepth": {
        "title": "CART, grown to pure leaves, no pruning, no depth limit",
        "impls": {
            "tree": ["--cart"],
        },
    },
    # Growing only, to depth 10 (the profiling benchmark, cluster-profiling; the
    # test-set benchmarks; cluster-cpu-vs-others with scikit-learn, rpart, J48
    # and YaDT).
    "cart_depth10": {
        "title": "CART, grown to depth 10, no pruning",
        "impls": {
            "tree": ["--cart", "-d", "10"],
            "sklearn": {"mode": "fit", "max_depth": 10},
            "rpart": {"mode": "fit", "maxdepth": 10},
            # the GPU libraries' one tree (cluster-gpu-vs-others' gpubench.py;
            # run.py runs them only for the end-to-end benchmark)
            "cuml": {"depth": 10},
            "xgboost": {"depth": 10},
        },
    },
    "c45_depth10_unpruned": {
        "title": "C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)",
        "impls": {
            "tree": ["--c45", "-d", "10"],
            # J48 has no depth limit: the adapter's own (J48Fit max_depth).
            "j48": {"options": "-U -M 2", "max_depth": 10},
            # YaDT's own depth limit (root at depth 0, as ./tree's -d); -np: only
            # C4.5's pre-pruning, no error-based pruning.
            "yadt": ["-np", "-m", "2", "-depth", "10"],
        },
    },
    # Pruned at the same fixed values, to depth 10 (the test-set benchmarks:
    # cluster-higgs, -susy, -ecbdl).
    "cart_depth10_alpha": {
        "title": "CART, grown to depth 10, pruned at a fixed alpha (cost-complexity)",
        "impls": {
            "tree": ["--cart", "-d", "10", "--prune", "{alpha}"],
        },
    },
    "c45_depth10": {
        "title": "C4.5, grown to depth 10, error-based pruning (CF 0.25, subtree raising)",
        "impls": {
            "tree": ["--c45", "-d", "10", "--prune", "0.25"],
        },
    },
    "c45": {
        "title": "C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows",
        "impls": {
            "tree": ["--c45", "--prune", "0.25"],
            "j48": {"options": "-C 0.25 -M 2"},
            "yadt": ["-ebpg", "-c", "0.25", "-m", "2"],
        },
    },
    "c45_unpruned": {
        "title": "C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows",
        "impls": {
            "tree": ["--c45"],
            "j48": {"options": "-U -M 2"},
            "yadt": ["-np", "-m", "2"],
        },
    },
}

# Growing only, to other depths, ./tree only (hybrid-vs-gpu-only: 5 to 30, with
# depth 10 above and cart_full_nodepth / c45_unpruned for no limit).
for _depth in (5, 15, 20, 25, 30):
    PROTOCOLS[f"cart_depth{_depth}"] = {
        "title": f"CART, grown to depth {_depth}, no pruning",
        "impls": {"tree": ["--cart", "-d", str(_depth)]},
    }
    PROTOCOLS[f"c45_depth{_depth}_unpruned"] = {
        "title": f"C4.5, grown to depth {_depth}, unpruned (only C4.5's collapse of useless splits)",
        "impls": {"tree": ["--c45", "-d", str(_depth)]},
    }

# ./tree on the GPU (tree_cuda: --hybrid, tree_gpu_only: --gpu) and ./tree --simple run
# every protocol ./tree runs, with the same flags.
for _protocol in PROTOCOLS.values():
    if "tree" in _protocol["impls"]:
        _protocol["impls"]["tree_cuda"] = _protocol["impls"]["tree"]
        _protocol["impls"]["tree_gpu_only"] = _protocol["impls"]["tree"]
        _protocol["impls"]["tree_simple"] = _protocol["impls"]["tree"]

# threads: "multi" = runs with every thread count asked for, "single" = only 1,
#   "all" = only the largest thread count asked for (./tree --hybrid: the CPU pool
#   that grows the small subtrees gets every CPU). ./tree --gpu grows
#   every node on the GPU, and its CPU threads feed the GPU and prune: "all" too.
# Every tool runs as one fresh process that fits exactly one tree, with no
#   warm-up fit before it (bench/README.md, "How time is measured").
IMPLS = {
    "tree": {"label": "./tree", "threads": "multi"},
    "tree_cuda": {"label": "./tree Hybrid", "threads": "all"},
    "tree_gpu_only": {"label": "./tree GPU", "threads": "all"},
    "tree_simple": {"label": "./tree simple", "threads": "single"},
    "sklearn": {"label": "scikit-learn", "threads": "single"},
    "rpart": {"label": "rpart", "threads": "single"},
    "j48": {"label": "Weka J48", "threads": "single"},
    "yadt": {"label": "YaDT", "threads": "multi"},
    "cuml": {"label": "cuML", "threads": "all"},
    "xgboost": {"label": "XGBoost", "threads": "all"},
}



# ------------------------------------------------------------------ datasets

def meta_path(dataset):
    """A dataset's meta.json: its own folder's, else <folder>/<dataset>.meta.json
    (a prefix_of set, or a set in a shared folder); None if it is not prepared."""
    own = DATA / dataset / "meta.json"
    if own.exists():
        return own
    return next(iter(sorted(DATA.glob(f"*/{dataset}.meta.json"))), None)


def prepared():
    """The names of every prepared dataset (prefix_of sets included)."""
    names = {d.name for d in DATA.iterdir() if (d / "meta.json").exists()}
    names |= {f.name.removesuffix(".meta.json") for f in DATA.glob("*/*.meta.json")}
    return sorted(names)


def meta(dataset):
    path = meta_path(dataset)
    if path is None:
        raise FileNotFoundError(f"{dataset}: not prepared (bench/prepare.py {dataset})")
    return json.load(open(path))


def csv_names(m):
    """The feature and class names train.csv uses: f<i> and c<k>, or a linked
    source's own (meta.json csv_features / csv_classes)."""
    return (m.get("csv_features", m["features"]), m.get("csv_classes", m["classes"]))


def stem(dataset, part):
    """The path of a dataset's `part` rows without the extension (.f32, .y.i32,
    .csv, .yadt.csv): <folder>/train or test in its own folder, <folder>/<name>
    (training rows only) in a shared one; a prefix_of set's parent's."""
    m = meta(dataset)
    if "prefix_of" in m:
        return stem(m["prefix_of"], part)
    path = meta_path(dataset)
    if path.name == "meta.json":
        return path.parent / part
    if part != "train":
        raise ValueError(f"{dataset}: a set in a shared folder has training rows only")
    return path.parent / dataset


def folder(dataset):
    """The folder of the dataset's files: its parent's for a prefix_of set."""
    return stem(dataset, "train").parent


def binary_meta(path):
    """The meta.json ./tree reads for <stem>.f32: <stem>.meta.json if there is
    one, else the folder's meta.json (src/core/dataset.cpp, loadBinary)."""
    path = Path(path)
    own = path.with_suffix(".meta.json")
    return own if own.exists() else path.parent / "meta.json"


def tree_input(dataset):
    """./tree's arguments for the training rows: the binary dump (<stem>.f32,
    with <stem>.y.i32 and the meta.json beside it; the same values as the CSV,
    loaded without parsing text), after --rows N for a prefix_of set (the
    first N rows of its parent's)."""
    m = meta(dataset)
    rows = ["--rows", str(m["n_train"])] if "prefix_of" in m else []
    return rows + [f"{stem(dataset, 'train')}.f32"]


def load(dataset, part):
    m = meta(dataset)
    rows, F = m[f"n_{part}"], len(m["features"])
    if rows == 0:  # no test set
        return np.empty((0, F), "<f4"), np.empty(0, "<i4")
    # count: a prefix_of set is the first rows of its parent's files
    X = np.fromfile(f"{stem(dataset, part)}.f32", dtype="<f4", count=rows * F)
    y = np.fromfile(f"{stem(dataset, part)}.y.i32", dtype="<i4", count=rows)
    if len(y) != rows or X.size != rows * F:
        raise AssertionError(f"{dataset} {part}: the files have fewer than {rows} rows")
    return X.reshape(rows, F), y


def check_csv(path, X, y, header, classes=None, prefix=False):
    """Every row of a prepared CSV (prefix: its first len(y) rows, for a
    prefix_of set) must read back as exactly the binary files' float32 values
    and class indices (labels `classes`, by default c0..c{K-1}). Raises
    otherwise. Read in batches: a whole SOREL CSV held about 135 GiB."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    F = X.shape[1]
    names = [f"f{j}" for j in range(F)] + ["class"]
    # 64-bit offsets: 2G rows of "c0" are more than 2^31 bytes (chess_2b).
    classes = pa.array(classes or [f"c{k}" for k in range(int(y.max()) + 1)], pa.large_string())
    reader = pacsv.open_csv(
        path, read_options=pacsv.ReadOptions(column_names=names, skip_rows=int(header),
                                             block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(
            column_types={**{f"f{j}": pa.float32() for j in range(F)}, "class": pa.string()}))
    done = extra = 0  # rows checked; rows past len(y)
    for whole in reader:
        if done == len(y) and prefix:
            break
        batch = whole.slice(0, len(y) - done)
        extra += whole.num_rows - batch.num_rows
        rows = slice(done, done + batch.num_rows)
        for j in range(F):
            if not np.array_equal(batch.column(j).to_numpy(), X[rows, j]):
                raise AssertionError(f"{path}: column f{j} differs from the binary file")
        if not pc.all(pc.equal(batch.column(F), pc.take(classes, pa.array(y[rows])))).as_py():
            raise AssertionError(f"{path}: class labels differ from the binary file")
        done += batch.num_rows
    if done != len(y) or (extra and not prefix):
        raise AssertionError(f"{path}: {done + extra} rows, expected {len(y)}")
    del reader
    pa.default_memory_pool().release_unused()


def csv_prefix_bytes(path, rows):
    """The bytes of a CSV's header and first `rows` lines (what ./tree --rows reads)."""
    size, lines = 0, rows + 1
    with open(path, "rb") as handle:
        while block := handle.read(1 << 24):
            count = block.count(b"\n")
            if count < lines:
                size, lines = size + len(block), lines - count
                continue
            end = -1
            for _ in range(lines):
                end = block.index(b"\n", end + 1)
            return size + end + 1
    raise AssertionError(f"{path}: fewer than {rows} rows")


def verify_dataset(dataset):
    """Prove that every tool reads the same data: SHA-256 of every file of the
    prepared dataset (of a prefix_of set: the bytes of its parent's files that
    the tools read), and every CSV row checked against the binary files.
    Returns {file name: sha256}."""
    import hashlib
    m = meta(dataset)
    F = len(m["features"])
    train = stem(dataset, "train")
    limits = {}  # file -> bytes read; the whole file if absent
    if "prefix_of" in m:
        limits = {Path(f"{train}.csv"): csv_prefix_bytes(f"{train}.csv", m["n_train"]),
                  Path(f"{train}.f32"): 4 * F * m["n_train"], Path(f"{train}.y.i32"): 4 * m["n_train"]}
    files = [meta_path(dataset), folder(dataset) / "yadt.names"]
    for part in ("train", "test") if m["n_test"] else ("train",):
        files += [Path(f"{stem(dataset, part)}{ext}") for ext in (".f32", ".y.i32", ".csv", ".yadt.csv")]
    hashes = {}
    for path in files:
        if not path.exists():
            continue
        digest = hashlib.sha256()
        left = limits.get(path, float("inf"))
        with open(path, "rb") as handle:
            while left > 0 and (chunk := handle.read(int(min(1 << 24, left)))):
                digest.update(chunk)
                left -= len(chunk)
        hashes[path.name] = digest.hexdigest()
    for part in ("train", "test"):
        X, y = load(dataset, part)
        if not len(y):
            continue
        prefix = part == "train" and "prefix_of" in m
        yadt = Path(f"{stem(dataset, part)}.yadt.csv")
        if yadt.exists():  # not written for huge datasets
            check_csv(yadt, X, y, header=False, prefix=prefix)
        if part == "train":
            check_csv(f"{train}.csv", X, y, header=True, classes=csv_names(m)[1], prefix=prefix)
    return hashes


# ------------------------------------------------------------------ ./tree dumps

def parse_tree_dump(path, features, classes):
    """./tree --dump -> flat preorder arrays (feature, threshold, value, left, right);
    leaves have feature -1. features, classes: the names train.csv uses (csv_names)."""
    import re
    node_re = re.compile(r"^\s*\w+ \[n=\d+\]: (.*)$")
    feature_index = {name: j for j, name in enumerate(features)}
    class_index = {name: k for k, name in enumerate(classes)}
    feature, threshold, value = [], [], []
    for line in open(path):
        match = node_re.match(line)
        if not match:
            continue
        body = match.group(1)
        if body.startswith("Leaf -> "):
            feature.append(-1)
            threshold.append(0.0)
            value.append(class_index[body[len("Leaf -> "):]])
        else:
            # The names as ./tree read them from the header: Arrow quotes f<i>.
            name, cut = re.match(r"if (\S+) <= (\S+)$", body).groups()
            feature.append(feature_index[name.strip('"')])
            threshold.append(float(cut))
            value.append(-1)
    feature = np.array(feature)
    count = len(feature)
    size = np.ones(count, dtype=np.int64)
    left = np.full(count, -1)
    right = np.full(count, -1)
    # Preorder: the left child follows its parent, the right child follows the
    # left subtree; one backwards pass gives subtree sizes.
    for node in range(count - 1, -1, -1):
        if feature[node] >= 0:
            left[node] = node + 1
            right[node] = node + 1 + size[node + 1]
            size[node] = 1 + size[left[node]] + size[right[node]]
    return feature, np.array(threshold), np.array(value), left, right


def predict_tree(tree, X):
    feature, threshold, value, left, right = tree
    node = np.zeros(len(X), dtype=np.int64)
    active = np.arange(len(X))
    while len(active):
        f = feature[node[active]]
        inner = f >= 0
        active = active[inner]
        if not len(active):
            break
        current = node[active]
        goes_left = X[active, f[inner]].astype(np.float64) <= threshold[current]
        node[active] = np.where(goes_left, left[current], right[current])
    return value[node]


# ------------------------------------------------------------------ hardware

def _cpu_list_count(text):
    count = 0
    for part in (text or "").split(","):
        if "-" in part:
            first, last = map(int, part.split("-"))
            count += last - first + 1
        elif part:
            count += 1
    return count


def hardware(machine):
    """machine.json of a run -> short descriptions: {"cpu", "ram", "gpu"} (gpu
    None when the run recorded none), e.g. "Intel Core i7-14700KF, 8P + 12E
    cores, 28 threads" and "32 GiB DDR4-3200 (2 × 16 GiB)"."""
    import re
    model = re.search(r"Model name:\s+(.*)", machine.get("lscpu", ""))
    model = re.sub(r"\(R\)|\(TM\)|\d+th Gen |\s+CPU.*|\s+\d+-Core Processor", "",
                   model.group(1)).strip() \
        if model else "unknown CPU"
    if model.startswith("INTEL "):  # "INTEL XEON GOLD 6542Y" -> "Intel Xeon Gold 6542Y"
        model = " ".join(w.capitalize() if w.isalpha() else w for w in model.split())
    p_logical = _cpu_list_count(machine.get("p_cores"))
    e_cores = _cpu_list_count(machine.get("e_cores"))
    sockets = re.search(r"Socket\(s\):\s+(\d+)", machine.get("lscpu", ""))
    per_socket = re.search(r"Core\(s\) per socket:\s+(\d+)", machine.get("lscpu", ""))
    if p_logical:
        p_cores = p_logical // 2 if machine.get("smt_active") == "1" else p_logical
        cores = f"{p_cores}P + {e_cores}E cores, "
    elif sockets and per_socket and int(sockets.group(1)) > 1:
        model = f"{sockets.group(1)}× {model}"
        cores = f"{sockets.group(1)} × {per_socket.group(1)} cores, "
    else:
        cores = ""
    cpu = f"{model}, {cores}{machine.get('logical_cpus')} threads"
    total = f"{machine['mem_total_kb'] / 2**20:.0f} GiB" if machine.get("mem_total_kb") else "?"
    modules = machine.get("ram_modules") or []
    if modules:
        kinds = sorted({(m["type"], m["mts"]) for m in modules})
        sizes = sorted({m["bytes"] for m in modules})
        kind = ", ".join(f"{t}-{mts}" for t, mts in kinds)
        layout = (f" ({len(modules)} × {sizes[0] / 2**30:.0f} GiB)" if len(sizes) == 1 else "")
        ram = f"{sum(m['bytes'] for m in modules) / 2**30:.0f} GiB {kind}{layout}"
        if sum(m["bytes"] for m in modules) < machine.get("mem_total_kb", 0) * 1024:
            # Fewer modules than the memory the kernel has: udev's DMI list is cut
            # short on boards with many slots (e.g. 24), so give the kernel's total.
            ram = f"{total} {kind}"
    else:
        ram = f"{total} RAM"
    gpu = (machine.get("versions") or {}).get("gpu") or ""
    if gpu and not gpu.startswith("unavailable"):
        fields = [f.strip() for f in gpu.splitlines()[0].split(",")]
        mib = re.match(r"(\d+)", fields[1]) if len(fields) > 1 else None
        vram = f", {round(int(mib.group(1)) / 1024)} GB VRAM" if mib else ""
        gpu = fields[0].replace("NVIDIA GeForce ", "NVIDIA ") + vram
    else:
        gpu = None
    return {"cpu": cpu, "ram": ram, "gpu": gpu}


def failure(row):
    """A failed run's reason, short enough for a chart label."""
    error = row.get("error") or ""
    if row["status"] == "timeout":
        return "timeout"
    if "OutOfMemoryError" in error:
        return "out of memory (Java heap)"
    if row["status"] == "error" and not error and row.get("peak_rss_bytes"):
        # no output at all: killed by a signal, on these nodes the kernel's OOM killer
        return f"killed at {int(row['peak_rss_bytes']) / 2**30:,.0f} GiB RAM (out of memory)"
    return row["status"]


def hardware_line(machine, gpu_used):
    """One line for a chart: "CPU ... · RAM ... · GPU ..." (or "no GPU used")."""
    info = hardware(machine)
    gpu = f"GPU {info['gpu']}" if gpu_used and info["gpu"] else "no GPU used"
    return f"CPU {info['cpu']}  ·  RAM {info['ram']}  ·  {gpu}"


# ---------------------------------------------------------------- out of date
#
# The final protocol (BENCHMARK-CHANGES.MD) starts at these commits. A result
# from an older build is out of date: every chart draws it faded and labelled
# as a placeholder until the rerun replaces it, and the label goes away by
# itself once the chart is redrawn from the new results. GPU runs (hopper,
# ampere): one GPU timing rule, an 80 GB GPU, hand-off buffers only for
# --hybrid, and ./tree without --gpu-startup (only the CUDA context untimed,
# 17d8b2f3; cuML and XGBoost on the GPU nodes: 0cff6d1b, as 17d8b2f3 changed
# only ./tree); every other run: the final harness (03c4e2d2). A run on
# dell1 is out of date too: the paper's dell is a 512 GB node (dell2-4, 8 of 16
# memory channels); dell1 has 1 TB and every channel, 1.8x the bandwidth.
FINAL_SINCE = {"cpu": "03c4e2d2", "gpu": "17d8b2f3", "gpulibs": "0cff6d1b"}
GPU_LIBRARIES = ("cuml", "xgboost")
GPU_HOSTS = ("hopper", "ampere")
RETIRED_HOSTS = ("dell1",)
OUT_OF_DATE_NOTE = (f"faded, ⚠ out of date: measured before the final protocol (./tree's GPU runs before {FINAL_SINCE['gpu']}, cuML/XGBoost before {FINAL_SINCE['gpulibs']}, "
                    f"CPU runs before {FINAL_SINCE['cpu']}) or on dell1 (1 TB; the final dell is a 512 GB node); "
                    "a placeholder until the rerun")


@functools.lru_cache(maxsize=None)
def _built_on(commit, base):
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", base, commit],
                          capture_output=True).returncode == 0


def out_of_date(commit, host, impl=None):
    """True if a result of this build (a commit, `git describe` or not) on this host is from
    before the final protocol, from a retired host (dell1), or its build is unknown."""
    commit = (commit or "").strip().removesuffix("-dirty")
    commit = commit.rsplit("-g", 1)[-1] if "-g" in commit else commit
    node = (host or "").split(".")[0]
    if node in RETIRED_HOSTS:
        return True
    kind = "cpu" if node not in GPU_HOSTS else "gpulibs" if impl in GPU_LIBRARIES else "gpu"
    return not commit or not _built_on(commit, FINAL_SINCE[kind])


def rows_out_of_date(rows, machine=None):
    """True if any of a bar's rows is out of date: each row's commit and host, else its
    machine.json's (a run folder's: versions.tree_git and host; gpubench's: commit and host)."""
    machine = machine or {}
    commit = (machine.get("versions") or {}).get("tree_git") or machine.get("commit")
    host = machine.get("host")
    rows = list(rows)
    if not rows:
        return out_of_date(commit, host)
    return any(out_of_date(r.get("commit") or commit, r.get("host") or host, r.get("impl")) for r in rows)
