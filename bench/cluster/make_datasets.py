#!/usr/bin/env python3
"""Rebuild datasets/{covertype,higgs,supersymmetry}.csv from public downloads,
byte for byte as the copies on the desktop, make datasets/airline.csv,
datasets/ecbdl.csv and datasets/chess_*.csv, and check them all by SHA-256.

  bench/.venv/bin/python -I bench/cluster/make_datasets.py ~/downloads [names...]

Sources, in <downloads>:
  covertype.zip            UCI 31 (covtype.data.gz): same rows and order
  higgs_openml_45570.pq    OpenML 45570 = UCI HIGGS, same rows and order; its
                           doubles are the float32 values of UCI's text, parsed
                           slightly off, so they are rounded back to float32 and
                           printed as UCI does (%.18e)
  susy.zip                 UCI 279 (SUSY.csv.gz): the same float32 values and
                           labels, but the desktop's rows are shuffled (no
                           common seed reproduces it) and its doubles are UCI's
                           text parsed by an unknown reader, a few ulps off the
                           float32 values, written with repr. Neither is public,
                           so two files made from the desktop copy carry them:
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
                           (2026-10-19), so datasets/chess_*.csv are the only copies;
                           the originals' SHA-256 are in bench/cluster/README.md

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
# sha256sum of the desktop's datasets/ (2026-10-14); airline, ecbdl: of their first build.
EXPECTED = {
    "covertype": "fd7a387defeab82ea260a5be0c42b84e3426b963bb7feb490ffbafdaf89dac77",
    "higgs": "71f566a24e8a03ff51aa6961addc652d44ff8878b28f4198a7f8921f3c6af31c",
    "supersymmetry": "85ecf105d7667d88006de9d8dfafabbc703647fe651a40df51f9a1202d31108e",
    "airline": "255426039068522c2a5fb780b6fe4b31179bdc8988013d15db22ba914a742bcb",
    "ecbdl": "5bc39aecbec956ccb06b77295e292e0166db5f87c8341678b1e0ca35fd67120c",
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
    features, labels = features[order], labels[order]
    delta = np.fromfile(downloads / "susy_delta.i8", np.int8).astype(np.int64).reshape(-1, 18)
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


def chess_csv(path, rows, out):
    """One C4.5 file as ./tree's CSV: a header, values as written, the class
    without C4.5's closing period. The sources mix their line ends (CRLF from
    10M on), end with DOS's Ctrl-Z (100M; 1000M after each of its ten copies)
    and space the test rows of 10M and up ("0.182, 0.025, B.")."""
    out.write("x,y,class\n")
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
            out.write(text.decode("ascii"))
            count += lines
            if not block:
                break
    assert count == rows, f"{path}: {count:,} rows, expected {rows:,}"


def chess(name, part, downloads, out):
    source, train_rows, test_rows = CHESS[name]
    extension, rows = (".data", train_rows) if part == "train" else (".test", test_rows)
    chess_csv(downloads / "chess" / source / (source + extension), rows, out)


BUILDERS = {"covertype": covertype, "higgs": higgs, "supersymmetry": supersymmetry,
            "airline": airline, "ecbdl": ecbdl,
            **{name: partial(chess, name, "train") for name in CHESS},
            **{f"{name}_test": partial(chess, name, "test") for name in CHESS}}


def main():
    downloads = Path(sys.argv[1]).expanduser()
    names = sys.argv[2:] or list(BUILDERS)
    OUT.mkdir(exist_ok=True)
    failed = False
    for name in names:
        path = OUT / f"{name}.csv"
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
