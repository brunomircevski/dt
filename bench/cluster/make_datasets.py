#!/usr/bin/env python3
"""Rebuild datasets/{covertype,higgs}.csv from public downloads, byte for byte
as the copies on the desktop, make datasets/supersymmetry.csv (UCI's order),
datasets/supersymmetry_20k.csv (the desktop's first 20,000 rows, for
tests/check.sh), datasets/airline.csv, datasets/ecbdl/ecbdl.csv and
datasets/chess/chess_*.csv, and check them all by SHA-256. On mordor3
datasets/ is a link to ~/datasets and the downloads are in ~/datasets/sources.

  bench/.venv/bin/python -I bench/cluster/make_datasets.py ~/datasets/sources [names...]

Sources, in <downloads>:
  covertype.zip            UCI 31 (covtype.data.gz): same rows and order
  higgs_openml_45570.pq    OpenML 45570 = UCI HIGGS, same rows and order; its
                           doubles are the float32 values of UCI's text, parsed
                           slightly off, so they are rounded back to float32 and
                           printed as UCI does (%.18e)
  susy.zip                 UCI 279 (SUSY.csv.gz): supersymmetry.csv is its text in
                           its order (since 2026-10-23), so the last 500,000 rows
                           are the published test set. The desktop's copy had the
                           same float32 values and labels, but its rows shuffled
                           (no common seed reproduces it) and its doubles were
                           UCI's text parsed by an unknown reader, a few ulps off
                           the float32 values, written with repr. Neither is
                           public, so two files made from the desktop copy carry
                           them (supersymmetry_20k, the golden trees' sample):
  susy_order.u64           row order: blake2b-64 of each desktop row's float32
                           features and label byte (1 = signal), in file order
  susy_delta.i8            every value's double minus its float32, in float64
                           ulps, row-major in the desktop's order
  airline_14col.data.bz2   US flights 1987-2008 as in gbm-bench and the XGBoost
                           GPU paper: 115M rows, 13 features, from
                           https://kt.ijs.si/elena_ikonomovska/datasets/airline/
                           (SHA-256 1f13460f...). There is no desktop copy: the
                           expected SHA-256 is that of the first build (mordor3, 2026-10-14)
  ecbdl/training_set.arff  ECBDL'14 (protein contact maps, the GECCO 2014 big data
                           competition): 31,992,921 rows, 631 features, label 0/1.
                           On mordor3 a link to /home/Datasets/ECBDL (60.8 GB).
                           The expected SHA-256 is that of the first build (mordor3, 2026-10-16)
  chess/chess3x3x*/        a 3x3 checkerboard in Quinlan's C4.5 format (.names, .data,
                           .test): 1M, 10M, 100M and 1000M rows. Not public: it was
                           in mordor3's ~/chess and was deleted after the first build
                           (2026-10-19), so datasets/chess/chess_*.csv are the only copies;
                           the originals' SHA-256 are in bench/cluster/README.md.
                           Written in ./tree's format (f0,f1 and c0/c1, as prepare.py's
                           train.csv) since 2026-10-22 (chess_tree_format.py)
  (none)                   chess_2b: chess_100m's rows, then 1.9G rows drawn from a fixed seed

A file whose SHA-256 differs from the expected one is left as <name>.csv.mismatch
and the script fails.
"""

import gzip
import hashlib
import io
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets"
# sha256sum of the desktop's datasets/ (2026-10-14); airline, ecbdl, chess: of their first build.
EXPECTED = {
    "covertype": "fd7a387defeab82ea260a5be0c42b84e3426b963bb7feb490ffbafdaf89dac77",
    "higgs": "71f566a24e8a03ff51aa6961addc652d44ff8878b28f4198a7f8921f3c6af31c",
    # UCI's order (2026-10-23, first build); the desktop's whole file was 85ecf105...,
    # its first 20,000 rows (head -n 20001) are supersymmetry_20k.
    "supersymmetry": "d152c7592b4bec4b025931eccdf45b3900f1788903bedbea1332b1ca05b04a24",
    "supersymmetry_20k": "4563a7385202d9e65dca12b36f5c5c0f66967751df2acf73fe100353f297376b",
    "airline": "255426039068522c2a5fb780b6fe4b31179bdc8988013d15db22ba914a742bcb",
    "ecbdl": "5bc39aecbec956ccb06b77295e292e0166db5f87c8341678b1e0ca35fd67120c",
    # chess: ./tree's format (2026-10-22, chess_tree_format.py) of the first build
    # (mordor3, 2026-10-19), from files that no longer exist; chess_2b: its first build.
    "chess_1m": "e627a390685d94edf73cec0fe90ff3bfcdbda6e4fb0f20c9bfb33eb895cee18a",
    "chess_1m_test": "4fcfc226655dec9b32b021f7936dfdeb61e8311413d7d58466cd47b8d060bfee",
    "chess_10m": "68d6c5d4fc84f059955255f6197a26863f356161433774a43d1028e1f4d9443c",
    "chess_10m_test": "1feb48d239d7534f47d3d7bdbbacfddbdc062c21794156a7a78f38578605704d",
    "chess_100m": "944a3be648eb70de844473dc64abf164b84e56de67eec7b0c209b4caf0b630fe",
    "chess_100m_test": "1feb48d239d7534f47d3d7bdbbacfddbdc062c21794156a7a78f38578605704d",
    "chess_1b": "124073b81a26e28a693ce3ce4a5e3ccf8edc41d8dcdcd7a2431fb642fb42fca1",
    "chess_1b_test": "4fcfc226655dec9b32b021f7936dfdeb61e8311413d7d58466cd47b8d060bfee",
    "chess_2b": "9b4e47c7f391a0cfeec85b98e42f520461aba03b1afcbcbec783ceba9220add3",
}

COVERTYPE_HEADER = ",".join(
    ["Id", "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
     "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am",
     "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
    + [f"Wilderness_Area_{i}" for i in range(1, 5)]
    + [f"Soil_Type_{i}" for i in range(1, 41)] + ["CoverType"])
COVER_TYPES = {"1": "Spruce_Fir", "2": "Lodgepole_Pine", "3": "Ponderosa_Pine",
               "4": "Cottonwood_Willow", "5": "Aspen", "6": "Douglas_Fir", "7": "Krummholz"}

HIGGS_HEADER = ("Id,lepton_pT,lepton_eta,lepton_phi,missing_energy_magnitude,missing_energy_phi,"
                "jet_1_pT,jet_1_eta,jet_1_phi,jet_1_b-tag,jet_2_pT,jet_2_eta,jet_2_phi,jet_2_b-tag,"
                "jet_3_pT,jet_3_eta,jet_3_phi,jet_3_b-tag,jet_4_pT,jet_4_eta,jet_4_phi,jet_4_b-tag,"
                "m_jj,m_jjj,m_lv,m_jlv,m_bb,m_wbb,m_wwbb,Label")
CHUNK = 250_000


def covertype(downloads, out):
    with zipfile.ZipFile(downloads / "covertype.zip") as archive, \
            gzip.open(archive.open("covtype.data.gz"), "rt") as rows:
        out.write(COVERTYPE_HEADER + "\n")
        for i, line in enumerate(rows, 1):
            features, label = line.rstrip("\n").rsplit(",", 1)
            out.write(f"{i},{features},{COVER_TYPES[label]}\n")


def _higgs_chunk(job):
    first_id, values, labels = job
    text = io.StringIO()
    np.savetxt(text, values.astype(np.float32).astype(np.float64), fmt="%.18e", delimiter=",")
    lines = text.getvalue().splitlines()
    return "".join(f"{first_id + i},{line},{label}\r\n"
                   for i, (line, label) in enumerate(zip(lines, labels)))


def higgs(downloads, out):
    import pyarrow.parquet as pq
    table = pq.read_table(downloads / "higgs_openml_45570.pq")
    target = table.column("Target").to_pylist()
    assert set(target) == {"0.0", "1.0"}, set(target)
    labels = ["signal" if t == "1.0" else "background" for t in target]
    values = np.column_stack([table.column(i).to_numpy() for i in range(1, table.num_columns)])
    assert values.shape == (11_000_000, 28), values.shape
    del table, target
    jobs = ((start + 1, values[start:start + CHUNK], labels[start:start + CHUNK])
            for start in range(0, len(values), CHUNK))
    out.write(HIGGS_HEADER + "\r\n")
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        for text in pool.map(_higgs_chunk, jobs):
            out.write(text)


SUSY_HEADER = ("Id,lepton_1_pT,lepton_1_eta,lepton_1_phi,lepton_2_pT,lepton_2_eta,lepton_2_phi,"
               "missing_energy_magnitude,missing_energy_phi,MET_rel,axial_MET,M_R,M_TR_2,R,MT2,"
               "S_R,M_Delta_R,dPhi_r_b,costheta_r1,Label")


def susy_rows(features, labels):
    """float32 features (n x 18) and 0/1 labels -> one blake2b-64 per row (susy_order.u64)."""
    rows = np.concatenate([features.view(np.uint8).reshape(len(features), 72),
                           labels.astype(np.uint8)[:, None]], axis=1)
    return np.fromiter((int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(),
                                       "little") for r in rows), dtype="<u8", count=len(rows))


def susy_text(features, delta):
    """float32 features and their ulp offsets -> the doubles, as text."""
    exact = (features.astype(np.float64).view(np.int64) + delta).view(np.float64)
    return [",".join(map(repr, row)) for row in exact.tolist()]


def _susy_chunk(job):
    first_id, features, delta, labels = job
    return "".join(f"{first_id + i},{text},{label}\n" for i, (text, label)
                   in enumerate(zip(susy_text(features, delta), labels)))


def supersymmetry(downloads, out):
    """UCI's rows in UCI's order and text, label (first there) last as
    signal/background, an Id column first."""
    names = {"1.000000000000000000e+00": "signal", "0.000000000000000000e+00": "background"}
    out.write(SUSY_HEADER + "\n")
    rows = 0
    with zipfile.ZipFile(downloads / "susy.zip") as archive, \
            gzip.open(archive.open("SUSY.csv.gz"), "rt") as lines:
        for rows, line in enumerate(lines, 1):
            label, features = line.rstrip("\n").split(",", 1)
            assert features.count(",") == 17, rows
            out.write(f"{rows},{features},{names[label]}\n")
    assert rows == 5_000_000, rows


def supersymmetry_20k(downloads, out, rows=20_000):
    """The desktop's first `rows` rows, byte for byte: the sample of
    tests/check.sh's golden trees."""
    import pyarrow.csv as pacsv
    with zipfile.ZipFile(downloads / "susy.zip") as archive, \
            gzip.open(archive.open("SUSY.csv.gz")) as f:
        table = pacsv.read_csv(f, read_options=pacsv.ReadOptions(autogenerate_column_names=True),
                               convert_options=pacsv.ConvertOptions(
                                   column_types={f"f{i}": "float32" for i in range(19)}))
    uci = np.column_stack([table.column(i).to_numpy() for i in range(19)])
    labels, features = uci[:, 0], np.ascontiguousarray(uci[:, 1:])
    assert features.shape == (5_000_000, 18) and set(np.unique(labels)) == {0, 1}
    where = {h: i for i, h in enumerate(susy_rows(features, labels).tolist())}
    order = np.array([where[h] for h in np.fromfile(downloads / "susy_order.u64", "<u8").tolist()])
    assert np.array_equal(np.sort(order), np.arange(len(order))), "not a permutation"
    features, labels = features[order[:rows]], labels[order[:rows]]
    delta = np.fromfile(downloads / "susy_delta.i8", np.int8).astype(np.int64).reshape(-1, 18)
    delta = delta[:rows]
    names = np.where(labels == 1, "signal", "background")
    jobs = ((start + 1, features[start:start + CHUNK], delta[start:start + CHUNK],
             names[start:start + CHUNK]) for start in range(0, len(features), CHUNK))
    out.write(SUSY_HEADER + "\n")
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        for text in pool.map(_susy_chunk, jobs):
            out.write(text)


AIRLINE_COLUMNS = ["Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime", "CRSArrTime",
                   "UniqueCarrier", "FlightNum", "ActualElapsedTime", "Origin", "Dest",
                   "Distance", "Diverted", "ArrDelay"]
AIRLINE_CODES = ["UniqueCarrier", "Origin", "Dest"]


def airline(downloads, out):
    """gbm-bench's airline: the carrier and airports become integer codes in
    sorted order (pandas' category codes), the label is ArrDelay > 0. Rows in
    file order."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    stream = pa.input_stream(downloads / "airline_14col.data.bz2", compression="bz2")
    table = pacsv.read_csv(stream, read_options=pacsv.ReadOptions(column_names=AIRLINE_COLUMNS),
                           convert_options=pacsv.ConvertOptions(column_types={
                               name: pa.string() if name in AIRLINE_CODES else pa.int32()
                               for name in AIRLINE_COLUMNS}))
    assert not any(column.null_count for column in table.columns)
    for name in AIRLINE_CODES:
        encoded = pc.dictionary_encode(table[name].combine_chunks())
        rank = np.argsort(np.argsort(np.array(encoded.dictionary.to_pylist())))
        codes = rank.astype(np.int32)[encoded.indices.to_numpy()]
        table = table.set_column(table.column_names.index(name), name, pa.array(codes))
    delayed = pc.greater(table["ArrDelay"], 0)
    table = table.drop_columns(["ArrDelay"]).append_column(
        "Label", pc.if_else(delayed, "delayed", "on_time"))
    out.write(",".join(table.column_names) + "\n")  # Arrow would quote the names
    for batch in table.to_batches(max_chunksize=CHUNK * 4):
        text = pa.BufferOutputStream()
        pacsv.write_csv(batch, text, pacsv.WriteOptions(include_header=False,
                                                        quoting_style="none"))
        out.write(text.getvalue().to_pybytes().decode())


def ecbdl(downloads, out):
    """ECBDL'14 from its ARFF: 539 real attributes as float32 (the values
    prepare.py makes of them anyway), the 92 nominal ones (predicted secondary
    structure H/E/C, solvent accessibility 0-4; X = no residue at that window
    position) as integer codes of their values in sorted order, as airline's
    carrier and airports; label 0/1 (1 = contact) last. Rows in file order."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    path = downloads / "ecbdl" / "training_set.arff"
    names, nominal = [], {}
    with open(path, "rb") as arff:
        while line := arff.readline():
            text = line.decode().strip()
            if text.lower().startswith("@attribute"):
                _, name, kind = text.split(None, 2)
                names.append(name)
                if kind.startswith("{"):
                    nominal[name] = sorted(v.strip() for v in kind.strip("{}").split(","))
                else:
                    assert kind.lower() in ("real", "numeric", "integer"), text
            elif text.lower() == "@data":
                break
        label = names[-1]
        assert nominal.pop(label) == ["0", "1"], label
        assert (len(names), len(nominal)) == (632, 92), (len(names), len(nominal))
        sets = {name: pa.array(values) for name, values in nominal.items()}
        reader = pacsv.open_csv(  # from the first data row on
            arff, read_options=pacsv.ReadOptions(column_names=names, block_size=1 << 26),
            convert_options=pacsv.ConvertOptions(column_types={
                name: pa.string() if name in nominal or name == label else pa.float32()
                for name in names}))
        out.write(",".join(names) + "\n")
        rows = 0
        for batch in reader:
            columns = []
            for name, column in zip(names, batch.columns):
                assert not column.null_count, f"{name}: missing values"
                if name in sets:
                    column = pc.index_in(column, value_set=sets[name])
                    assert not column.null_count, f"{name}: a value outside {nominal[name]}"
                elif name == label:
                    assert pc.all(pc.is_in(column, value_set=pa.array(["0", "1"]))).as_py()
                columns.append(column)
            text = pa.BufferOutputStream()
            pacsv.write_csv(pa.record_batch(columns, names=names), text,
                            pacsv.WriteOptions(include_header=False, quoting_style="none"))
            out.write(text.getvalue().to_pybytes().decode())
            rows += batch.num_rows
    assert rows == 31_992_921, rows


# The chess sets: training rows (.data) and test rows (.test) of each source
# directory. 1000M is the 100M file ten times over (its rows repeat), and the
# 10M, 100M and 1000M sets share their test file.
CHESS = {"chess_1m": ("chess3x3x001M", 1_000_000, 1_000),
         "chess_10m": ("chess3x3x010M", 10_000_000, 4_500),
         "chess_100m": ("chess3x3x100M", 100_000_000, 4_500),
         "chess_1b": ("chess3x3x001MM", 1_000_000_000, 1_000)}
# The header as Arrow writes it: the chess CSVs are byte for byte the train.csv
# that bench/prepare.py would write, so it links to them instead (2026-10-22).
CHESS_HEADER = '"f0","f1","class"\n'


def chess_rows(x, y, b):
    """float32 x and y, True for class B -> ./tree's CSV rows, as bench/prepare.py
    writes them: Arrow's shortest text that reads back as the same float32, the
    class c0 (A) or c1 (B)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    labels = pc.take(pa.array(["c0", "c1"]), pa.array(b.astype(np.int8)))
    text = pa.BufferOutputStream()
    pacsv.write_csv(pa.table({"f0": x, "f1": y, "class": labels}), text,
                    pacsv.WriteOptions(include_header=False, quoting_style="none"))
    return text.getvalue().to_pybytes().decode("ascii")


def chess_text(text):
    """Rows "x,y,A|B\n" (C4.5's values) -> ./tree's CSV rows. The values are read
    as prepare.py reads them (double, then float32). They have at most six
    decimals and lie in [0, 4), where float32 is closer than half a millionth,
    so the original text follows from the float32 values (asserted)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    table = pacsv.read_csv(io.BytesIO(text), read_options=pacsv.ReadOptions(
        column_names=["x", "y", "class"]), convert_options=pacsv.ConvertOptions(
        column_types={"x": pa.float64(), "y": pa.float64(), "class": pa.string()}))
    x, y = table["x"].to_numpy(), table["y"].to_numpy()
    for v in (x, y):
        assert ((v >= 0) & (v < 4) & (np.rint(v * 1e6) / 1e6 == v)).all(), "not six decimals"
    b = pc.equal(table["class"], "B").to_numpy(zero_copy_only=False)
    return chess_rows(x.astype(np.float32), y.astype(np.float32), b)


def chess_csv(path, rows, out):
    """One C4.5 file as ./tree's CSV (chess_text). The sources mix their line
    ends (CRLF from 10M on), end with DOS's Ctrl-Z (100M; 1000M after each of
    its ten copies) and space the test rows of 10M and up ("0.182, 0.025, B.")."""
    out.write(CHESS_HEADER)
    count, tail = 0, b""
    with open(path, "rb") as source:
        while True:
            block = source.read(1 << 26)
            text = tail + block
            cut = len(text) if not block else text.rfind(b"\n") + 1
            text, tail = text[:cut], text[cut:]
            text = (text.replace(b"\x1a", b"").replace(b"\r", b"").replace(b" ", b"")
                    .replace(b".\n", b"\n"))
            if text and not text.endswith(b"\n"):
                assert not block, path  # only the last line may lack its newline
                text += b"\n"
            lines = text.count(b"\n")
            assert not text.translate(None, b"0123456789.,AB\n"), f"{path}: unexpected bytes"
            assert text.count(b",") == 2 * lines, f"{path}: not 3 fields per row"
            assert text.count(b"A\n") + text.count(b"B\n") == lines, f"{path}: class not A or B"
            assert b"\n\n" not in text and not text.startswith(b"\n"), f"{path}: empty line"
            if lines:
                out.write(chess_text(text))
            count += lines
            if not block:
                break
    assert count == rows, f"{path}: {count:,} rows, expected {rows:,}"


def chess(name, part, downloads, out):
    source, train_rows, test_rows = CHESS[name]
    extension, rows = (".data", train_rows) if part == "train" else (".test", test_rows)
    chess_csv(downloads / "chess" / source / (source + extension), rows, out)


# Generated chess sets (2026-10-22): the rows of a smaller set first (so the sizes
# nest), then new rows drawn as its rows were, CHESS_CHUNK per seed [CHESS_SEED, chunk].
CHESS_GENERATED = {"chess_2b": ("chess_100m", 2_000_000_000)}
CHESS_SEED = 20261022
CHESS_CHUNK = 10_000_000


def _chess_squares(micro):
    """B where the unit squares' indices add up to an even number (5 of the 9
    squares); 3.0 lies in the last square."""
    return np.minimum(micro // 1_000_000, 2).sum(axis=0) % 2 == 0


def chess_grid():
    """The values the first build's generator drew, in millionths: 3 * k / 32767
    for k = 0..32767 (a C rand() with RAND_MAX 32767) to six decimals. None lies
    on an inner square boundary."""
    return np.rint(np.arange(32768) * 3e6 / 32767).astype(np.int64)


def _chess_chunk(job):
    """x and y drawn uniformly from chess_grid(), as in the first build."""
    chunk, rows = job
    k = np.random.default_rng([CHESS_SEED, chunk]).integers(0, 32768, (2, rows))
    micro = chess_grid()[k]
    # A millionth count / 1e6 is the double nearest to the decimal, as parsing its text gives.
    return chess_rows((micro[0] / 1e6).astype(np.float32), (micro[1] / 1e6).astype(np.float32),
                      _chess_squares(micro))


def chess_check(path):
    """An existing set's rows must be drawn as _chess_chunk draws them."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(column_types={
        "f0": pa.float32(), "f1": pa.float32(), "class": pa.string()}))
    values = np.stack([table[f].to_numpy() for f in ("f0", "f1")])
    micro = np.rint(values.astype(np.float64) * 1e6).astype(np.int64)
    assert np.array_equal(values, (micro / 1e6).astype(np.float32)), "float32s _chess_chunk doesn't write"
    b = pc.equal(table["class"], "c1").to_numpy(zero_copy_only=False)
    print(f"  {path.name}: {table.num_rows:,} rows, {len(np.unique(micro[0])):,} and "
          f"{len(np.unique(micro[1])):,} distinct values, B {b.mean():.4%}")
    assert np.isin(micro, chess_grid()).all(), "values off the generator's grid"
    assert np.array_equal(b, _chess_squares(micro)), "another rule"


def chess_generated(name, downloads, out):
    prefix, rows = CHESS_GENERATED[name]
    path = csv_path(prefix)
    chess_check(path)
    out.write(CHESS_HEADER)
    digest, copied = hashlib.sha256(), 0
    with open(path, "rb") as source:
        while block := source.read(1 << 26):
            digest.update(block)
            text = block.decode("ascii")
            if not copied:
                assert text.startswith(CHESS_HEADER), path
                text = text[len(CHESS_HEADER):]
            out.write(text)
            copied += text.count("\n")
    assert digest.hexdigest() == EXPECTED[prefix], f"{path}: not the expected SHA-256"
    jobs = [(chunk, min(CHESS_CHUNK, rows - start))
            for chunk, start in enumerate(range(copied, rows, CHESS_CHUNK))]
    workers = len(os.sched_getaffinity(0))
    with ProcessPoolExecutor(workers) as pool:
        for first in range(0, len(jobs), 2 * workers):  # a bounded number of chunks in memory
            for text in pool.map(_chess_chunk, jobs[first:first + 2 * workers]):
                out.write(text)


BUILDERS = {"covertype": covertype, "higgs": higgs, "supersymmetry": supersymmetry,
            "supersymmetry_20k": supersymmetry_20k,
            "airline": airline, "ecbdl": ecbdl,
            **{name: partial(chess, name, "train") for name in CHESS},
            **{f"{name}_test": partial(chess, name, "test") for name in CHESS},
            **{name: partial(chess_generated, name) for name in CHESS_GENERATED}}


def csv_path(name):
    """chess_* in datasets/chess/, ecbdl in datasets/ecbdl/, the others in datasets/."""
    folder = "chess" if name.startswith("chess_") else "ecbdl" if name == "ecbdl" else ""
    return OUT / folder / f"{name}.csv"


def main():
    downloads = Path(sys.argv[1]).expanduser()
    names = sys.argv[2:] or list(BUILDERS)
    OUT.mkdir(exist_ok=True)
    failed = False
    for name in names:
        path = csv_path(name)
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            print(f"{path} exists, checking it")
        else:
            partial = path.with_suffix(".csv.partial")
            with open(partial, "w", newline="", buffering=1 << 24) as out:
                BUILDERS[name](downloads, out)
            partial.rename(path)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            while block := f.read(1 << 24):
                digest.update(block)
        expected = EXPECTED.get(name)
        same = digest.hexdigest() == expected
        print(f"{name}: {path.stat().st_size:,} bytes, sha256 {digest.hexdigest()} "
              + ("(nothing to compare with: record it in EXPECTED)" if expected is None
                 else "= the expected" if same else f"!= the expected {expected}"))
        if expected is not None and not same:
            path.rename(path.with_suffix(".csv.mismatch"))
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
