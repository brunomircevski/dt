"""Shared paths, dataset files and protocol definitions for the CPU benchmark.

Data layout (bench/data/<dataset>/, made by prepare.py):

  meta.json                  features, classes, row counts, how the split was made
  train.f32, test.f32        features, row-major little-endian float32 (n x F)
  train.y.i32, test.y.i32    class index per row, little-endian int32
  train.csv                  header f0..f{F-1},class; labels c0..c{K-1}  (./tree), or
                             a link to a source CSV with names of its own (ECBDL):
                             meta.json's csv_features / csv_classes
  yadt.names, train.yadt.csv, test.yadt.csv   headerless CSV + metadata  (YaDT)

Every tool sees the same float32 values: the CSVs print them exactly (9
significant digits), the binary files hold them as they are.
"""

import json
import os
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

# CART trees are capped at depth 30 for every tool: rpart cannot grow deeper,
# and a tree that only one tool may grow deeper is not the same work.
CART_DEPTH = 30

# The fixed alpha of cart_alpha (run.py --alpha overrides it). Every CART tool
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
            "tree": ["--cart", "--no-prune", "-d", str(CART_DEPTH)],
            "sklearn": {"mode": "fit", "max_depth": CART_DEPTH},
            "rpart": {"mode": "fit", "maxdepth": CART_DEPTH},
        },
    },
    "cart_depth12": {
        "title": "CART, grown to depth 12, no pruning",
        "impls": {
            "tree": ["--cart", "--no-prune", "-d", "12"],
            "sklearn": {"mode": "fit", "max_depth": 12},
            "rpart": {"mode": "fit", "maxdepth": 12},
        },
    },
    "cart_alpha": {
        "title": "CART, one tree pruned at a fixed alpha (cost-complexity)",
        "impls": {
            "tree": ["--cart", "-d", str(CART_DEPTH), "--alpha", "{alpha}"],
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
            "tree": ["--cart", "--alpha", "{alpha}"],
            "sklearn": {"mode": "fit", "ccp_alpha": "{alpha}"},
        },
    },
    "cart_full_nodepth": {
        "title": "CART, grown to pure leaves, no pruning, no depth limit",
        "impls": {
            "tree": ["--cart", "--no-prune"],
        },
    },
    # Growing only, to depth 10 (the profiling benchmark, cluster-profiling; the
    # test-set benchmarks; cluster-cpu-vs-others with scikit-learn and J48).
    "cart_depth10": {
        "title": "CART, grown to depth 10, no pruning",
        "impls": {
            "tree": ["--cart", "--no-prune", "-d", "10"],
            "sklearn": {"mode": "fit", "max_depth": 10},
        },
    },
    "c45_depth10_unpruned": {
        "title": "C4.5, grown to depth 10, unpruned (only C4.5's collapse of useless splits)",
        "impls": {
            "tree": ["--c45", "--no-prune", "-d", "10"],
            # J48 has no depth limit: the adapter's own (J48Fit max_depth).
            "j48": {"options": "-U -M 2", "max_depth": 10},
            # Not C4.5: scikit-learn's nearest tree, entropy, with C4.5's
            # smallest split (2 rows on each side).
            "sklearn": {"mode": "fit", "criterion": "entropy", "max_depth": 10,
                        "min_samples_leaf": 2},
        },
    },
    # Pruned at the same fixed values, to depth 10 (the test-set benchmarks:
    # cluster-airline, -higgs, -susy, -ecbdl).
    "cart_depth10_alpha": {
        "title": "CART, grown to depth 10, pruned at a fixed alpha (cost-complexity)",
        "impls": {
            "tree": ["--cart", "-d", "10", "--alpha", "{alpha}"],
        },
    },
    "c45_depth10": {
        "title": "C4.5, grown to depth 10, error-based pruning (CF 0.25, subtree raising)",
        "impls": {
            "tree": ["--c45", "-d", "10"],
        },
    },
    "c45": {
        "title": "C4.5, error-based pruning (CF 0.25, subtree raising), min 2 rows",
        "impls": {
            "tree": ["--c45"],
            "j48": {"options": "-C 0.25 -M 2"},
            "yadt": ["-ebpg", "-c", "0.25", "-m", "2"],
            # Not C4.5: scikit-learn's nearest tree, entropy, pruned the only
            # way it can, by cost-complexity at CART's alpha.
            "sklearn": {"mode": "fit", "criterion": "entropy", "ccp_alpha": "{alpha}"},
        },
    },
    "c45_unpruned": {
        "title": "C4.5, unpruned (only C4.5's collapse of useless splits), min 2 rows",
        "impls": {
            "tree": ["--c45", "--no-prune"],
            "j48": {"options": "-U -M 2"},
            "yadt": ["-np", "-m", "2"],
        },
    },
}

# ./tree on the GPU (--cuda, and --cuda --gpu-only) and ./tree --simple run
# every protocol ./tree runs, with the same flags.
for _protocol in PROTOCOLS.values():
    if "tree" in _protocol["impls"]:
        _protocol["impls"]["tree_cuda"] = _protocol["impls"]["tree"]
        _protocol["impls"]["tree_gpu_only"] = _protocol["impls"]["tree"]
        _protocol["impls"]["tree_simple"] = _protocol["impls"]["tree"]

# threads: "multi" = runs with every thread count asked for, "single" = only 1,
#   "all" = only the largest thread count asked for (./tree --cuda: the CPU pool
#   that grows the small subtrees gets every CPU). ./tree --cuda --gpu-only grows
#   every node on the GPU, and its CPU threads feed the GPU and prune: "all" too.
# warmup: True for the managed runtimes (Python, R, JVM): before the timed fit,
#   the adapter fits one untimed tree on the first WARMUP_ROWS training rows in
#   the same process (imports, first-call costs, JIT compilation). The native
#   executables need none: loading is not timed, so the file cache does not matter.
IMPLS = {
    "tree": {"label": "./tree", "threads": "multi", "warmup": False},
    "tree_cuda": {"label": "./tree CUDA", "threads": "all", "warmup": False},
    "tree_gpu_only": {"label": "./tree CUDA, GPU only", "threads": "all", "warmup": False},
    "tree_simple": {"label": "./tree simple", "threads": "single", "warmup": False},
    "sklearn": {"label": "scikit-learn", "threads": "single", "warmup": True},
    "rpart": {"label": "rpart", "threads": "single", "warmup": True},
    "j48": {"label": "Weka J48", "threads": "single", "warmup": True},
    "yadt": {"label": "YaDT", "threads": "multi", "warmup": False},
}

# Rows of the untimed warm-up fit (run.py --warmup-rows overrides it; "all" =
# the whole training set). The same number for every managed runtime.
WARMUP_ROWS = 50_000



# ------------------------------------------------------------------ datasets

def meta(dataset):
    return json.load(open(DATA / dataset / "meta.json"))


def csv_names(m):
    """The feature and class names train.csv uses: f<i> and c<k>, or a linked
    source's own (meta.json csv_features / csv_classes)."""
    return (m.get("csv_features", m["features"]), m.get("csv_classes", m["classes"]))


def load(dataset, part):
    m = meta(dataset)
    rows = m[f"n_{part}"]
    X = np.fromfile(DATA / dataset / f"{part}.f32", dtype="<f4").reshape(rows, len(m["features"]))
    y = np.fromfile(DATA / dataset / f"{part}.y.i32", dtype="<i4")
    return X, y


def check_csv(path, X, y, header, classes=None):
    """Every row of a prepared CSV must read back as exactly the binary files'
    float32 values and class indices (labels `classes`, by default c0..c{K-1}).
    Raises otherwise."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    F = X.shape[1]
    names = [f"f{j}" for j in range(F)] + ["class"]
    table = pacsv.read_csv(
        path, read_options=pacsv.ReadOptions(column_names=names, skip_rows=int(header)),
        convert_options=pacsv.ConvertOptions(
            column_types={**{f"f{j}": pa.float32() for j in range(F)}, "class": pa.string()}))
    if table.num_rows != len(y):
        raise AssertionError(f"{path}: {table.num_rows} rows, expected {len(y)}")
    for j in range(F):
        if not np.array_equal(table[f"f{j}"].to_numpy(), X[:, j]):
            raise AssertionError(f"{path}: column f{j} differs from the binary file")
    # 64-bit offsets: 2G rows of "c0" are more than 2^31 bytes (chess_2b).
    classes = pa.array(classes or [f"c{k}" for k in range(int(y.max()) + 1)], pa.large_string())
    if not pc.all(pc.equal(table["class"], pc.take(classes, pa.array(y)))).as_py():
        raise AssertionError(f"{path}: class labels differ from the binary file")
    del table
    pa.default_memory_pool().release_unused()  # ECBDL: tens of GB, not kept while tools run


def verify_dataset(dataset):
    """Prove that every tool reads the same data: SHA-256 of every file of the
    prepared dataset, and every CSV row checked against the binary files.
    Returns {file name: sha256}."""
    import hashlib
    folder = DATA / dataset
    hashes = {}
    for path in sorted(folder.iterdir()):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(1 << 24):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    for part in ("train", "test"):
        X, y = load(dataset, part)
        if (folder / f"{part}.yadt.csv").exists():  # not written for huge datasets
            check_csv(folder / f"{part}.yadt.csv", X, y, header=False)
        if part == "train":
            check_csv(folder / "train.csv", X, y, header=True, classes=csv_names(meta(dataset))[1])
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


def hardware_line(machine, gpu_used):
    """One line for a chart: "CPU ... · RAM ... · GPU ..." (or "no GPU used")."""
    info = hardware(machine)
    gpu = f"GPU {info['gpu']}" if gpu_used and info["gpu"] else "no GPU used"
    return f"CPU {info['cpu']}  ·  RAM {info['ram']}  ·  {gpu}"
