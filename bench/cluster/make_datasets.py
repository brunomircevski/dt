#!/usr/bin/env python3
"""Rebuild datasets/{covertype,higgs,supersymmetry}.csv from public downloads,
byte for byte as the copies on the desktop, make datasets/airline.csv, and
check them all by SHA-256.

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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "datasets"
# sha256sum of the desktop's datasets/ (2026-10-14); airline: of its first build.
EXPECTED = {
    "covertype": "fd7a387defeab82ea260a5be0c42b84e3426b963bb7feb490ffbafdaf89dac77",
    "higgs": "71f566a24e8a03ff51aa6961addc652d44ff8878b28f4198a7f8921f3c6af31c",
    "supersymmetry": "85ecf105d7667d88006de9d8dfafabbc703647fe651a40df51f9a1202d31108e",
    "airline": "255426039068522c2a5fb780b6fe4b31179bdc8988013d15db22ba914a742bcb",
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


BUILDERS = {"covertype": covertype, "higgs": higgs, "supersymmetry": supersymmetry,
            "airline": airline}


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
        same = digest.hexdigest() == EXPECTED[name]
        print(f"{name}: {path.stat().st_size:,} bytes, sha256 {digest.hexdigest()} "
              f"{'= the expected' if same else '!= the expected ' + EXPECTED[name]}")
        if not same:
            path.rename(path.with_suffix(".csv.mismatch"))
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
