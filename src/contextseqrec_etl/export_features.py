from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Item-feature exporter.
#
# Produces a dense, model-aligned matrix ``[num_items + 1, F]`` consumed by
# ContextSeqRec's optional ``sampling.item_features_path``. Row 0 is the padding
# item and is always zero; row ``i`` (1..M) holds the features of the dense item
# id ``i`` used by the SASRec model.
#
# Alignment is the critical invariant. The dense id comes exclusively from the
# trusted ``dataset.pkl`` ``smap`` (which maps the global ``events.parquet`` sid
# to 1..M). We recover attributes through:
#
#     events.parquet: sid -> canonical_listing_id
#     items.parquet : canonical_listing_id -> attributes
#     dataset.pkl   : smap[sid] = dense_id
#
# We never index by ``listing_id_numeric`` (regional, not offset-adjusted).
#
# All transformations are FIXED (no fitted statistics): log1p on skewed
# magnitudes, explicit missing indicators, one-hot over a catalog-derived but
# order-stable vocabulary, and multi-hot for amenities. This keeps the export
# deterministic and free of validation/test leakage: nothing here depends on the
# split, on interaction counts, or on any label.

SCHEMA_VERSION = "contextseqrec-item-features-v1"

CATEGORICAL_COLUMNS = ("business_type", "usage_type", "unit_type", "market_segment")
NUMERIC_LOG_COLUMNS = ("price", "usable_areas", "total_areas")
NUMERIC_COUNT_COLUMNS = ("bedrooms", "bathrooms", "suites", "parking_spaces")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    """Order-stable hash of a parquet directory (names + per-file digests)."""
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_path(path: Path) -> str:
    return _sha256_tree(path) if path.is_dir() else _sha256_file(path)


def load_smap(dataset_path: Path) -> dict[int, int]:
    """Load only the ``smap`` mapping from a trusted dataset pickle.

    ``dataset.pkl`` is produced by this ETL and must be treated as trusted; we
    still avoid materializing the huge parallel event vectors by discarding
    every top-level value except ``smap`` as soon as the object is built. On a
    memory-constrained host prefer running this on a machine that can hold the
    pickle; the exporter itself keeps only ``smap`` afterwards.
    """
    with dataset_path.open("rb") as stream:
        data = pickle.load(stream)
    smap = data.get("smap")
    if not isinstance(smap, dict) or not smap:
        raise ValueError("dataset.pkl does not contain a non-empty 'smap'")
    del data
    dense = {int(sid): int(dense_id) for sid, dense_id in smap.items()}
    values = sorted(dense.values())
    if values != list(range(1, len(values) + 1)):
        raise ValueError("smap dense ids must be a contiguous range starting at 1")
    return dense


def sid_to_canonical(events_dir: Path) -> dict[int, str]:
    """Build the unique ``sid -> canonical_listing_id`` map from events.

    Each sid must resolve to exactly one canonical id. A canonical id may be
    shared by more than one sid (the rare cross-region duplicate), which is
    allowed: those sids simply receive identical attribute rows.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(events_dir), format="parquet")
    mapping: dict[int, str] = {}
    scanner = dataset.scanner(
        columns=["sid", "canonical_listing_id"], batch_size=1_000_000
    )
    for batch in scanner.to_batches():
        sids = batch.column("sid").to_pylist()
        cans = batch.column("canonical_listing_id").to_pylist()
        for sid, canonical in zip(sids, cans):
            if sid is None or canonical is None:
                continue
            sid = int(sid)
            previous = mapping.get(sid)
            if previous is None:
                mapping[sid] = canonical
            elif previous != canonical:
                raise ValueError(
                    f"sid {sid} maps to multiple canonical ids: {previous!r} and {canonical!r}"
                )
    if not mapping:
        raise ValueError("events.parquet produced no sid -> canonical mapping")
    return mapping


def _parse_amenities(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text in {"[]", "None", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(item).strip().upper() for item in parsed if str(item).strip()]


def _stable_vocabulary(values: pd.Series, limit: int | None = None) -> list[str]:
    counts = values.value_counts()
    # Sort by descending frequency then lexicographically for deterministic ties.
    ordered = sorted(counts.index, key=lambda key: (-int(counts[key]), str(key)))
    ordered = [str(value) for value in ordered]
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def build_feature_matrix(
    items: pd.DataFrame,
    smap: dict[int, int],
    sid_canonical: dict[int, str],
    amenities_top: int,
) -> tuple[np.ndarray, list[str]]:
    num_items = len(smap)
    items = items.drop_duplicates("canonical_listing_id").set_index("canonical_listing_id")

    # dense_id -> canonical, through the sid bridge.
    dense_to_canonical: dict[int, str] = {}
    missing_sid = 0
    for sid, dense_id in smap.items():
        canonical = sid_canonical.get(sid)
        if canonical is None:
            missing_sid += 1
            continue
        dense_to_canonical[dense_id] = canonical
    if missing_sid:
        raise ValueError(
            f"{missing_sid} smap sids were absent from events.parquet; inputs are inconsistent"
        )

    # Column plan (fixed, no fitted statistics).
    columns: list[str] = []
    for name in NUMERIC_LOG_COLUMNS:
        columns.append(f"log1p_{name}")
        columns.append(f"missing_{name}")
    for name in NUMERIC_COUNT_COLUMNS:
        columns.append(name)
        columns.append(f"missing_{name}")
    columns.append("latitude")
    columns.append("longitude")
    columns.append("missing_latitude")
    columns.append("missing_longitude")

    categorical_vocab: dict[str, list[str]] = {}
    for name in CATEGORICAL_COLUMNS:
        if name == "market_segment":
            continue
        if name in items.columns:
            vocab = _stable_vocabulary(items[name].dropna().astype(str))
            categorical_vocab[name] = vocab
            columns.extend(f"{name}={value}" for value in vocab)

    amenity_vocab: list[str] = []
    if "amenities" in items.columns:
        exploded = items["amenities"].map(_parse_amenities)
        flat = pd.Series([a for row in exploded for a in row], dtype="object")
        amenity_vocab = _stable_vocabulary(flat, limit=amenities_top) if len(flat) else []
        columns.extend(f"amenity={value}" for value in amenity_vocab)

    feature_index = {name: index for index, name in enumerate(columns)}
    matrix = np.zeros((num_items + 1, len(columns)), dtype=np.float32)

    def to_float(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    filled = 0
    for dense_id in range(1, num_items + 1):
        canonical = dense_to_canonical.get(dense_id)
        if canonical is None or canonical not in items.index:
            # Item present in smap but absent from catalog: leave zero-with-missing.
            for name in (*NUMERIC_LOG_COLUMNS, *NUMERIC_COUNT_COLUMNS):
                matrix[dense_id, feature_index[f"missing_{name}"]] = 1.0
            matrix[dense_id, feature_index["missing_latitude"]] = 1.0
            matrix[dense_id, feature_index["missing_longitude"]] = 1.0
            continue
        row = items.loc[canonical]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        filled += 1

        for name in NUMERIC_LOG_COLUMNS:
            number = to_float(row.get(name)) if name in items.columns else None
            if number is None or number < 0:
                matrix[dense_id, feature_index[f"missing_{name}"]] = 1.0
            else:
                matrix[dense_id, feature_index[f"log1p_{name}"]] = math.log1p(number)
        for name in NUMERIC_COUNT_COLUMNS:
            number = to_float(row.get(name)) if name in items.columns else None
            if number is None:
                matrix[dense_id, feature_index[f"missing_{name}"]] = 1.0
            else:
                matrix[dense_id, feature_index[name]] = number

        lat = to_float(row.get("lat_region")) if "lat_region" in items.columns else None
        lon = to_float(row.get("lon_region")) if "lon_region" in items.columns else None
        if lat is None or lat == 0.0:
            matrix[dense_id, feature_index["missing_latitude"]] = 1.0
        else:
            matrix[dense_id, feature_index["latitude"]] = lat
        if lon is None or lon == 0.0:
            matrix[dense_id, feature_index["missing_longitude"]] = 1.0
        else:
            matrix[dense_id, feature_index["longitude"]] = lon

        for name, vocab in categorical_vocab.items():
            value = row.get(name)
            key = f"{name}={value}"
            if value is not None and key in feature_index:
                matrix[dense_id, feature_index[key]] = 1.0

        if amenity_vocab:
            for amenity in _parse_amenities(row.get("amenities")):
                key = f"amenity={amenity}"
                position = feature_index.get(key)
                if position is not None:
                    matrix[dense_id, position] = 1.0

    if filled == 0:
        raise ValueError("no catalog rows matched the dense item ids")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains non-finite values")
    if matrix[0].any():
        raise ValueError("padding row must remain all zeros")
    return matrix, columns


def _write_atomic_npy(matrix: np.ndarray, output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"output already exists; pass --force to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False, suffix=".npy") as stream:
            temporary_name = stream.name
            np.save(stream, matrix, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def export_item_features(
    dataset_path: Path,
    source_dir: Path,
    output: Path,
    amenities_top: int,
    force: bool,
) -> dict[str, Any]:
    events_dir = source_dir / "events.parquet"
    items_dir = source_dir / "items.parquet"
    if not events_dir.exists():
        raise FileNotFoundError(events_dir)
    if not items_dir.exists():
        raise FileNotFoundError(items_dir)

    smap = load_smap(dataset_path)
    sid_canonical = sid_to_canonical(events_dir)
    items = pd.read_parquet(items_dir)
    if "canonical_listing_id" not in items.columns:
        raise ValueError("items.parquet is missing canonical_listing_id")
    items["market_segment"] = items.apply(
        lambda row: _market_segment(row.get("business_type"), row.get("usage_type")),
        axis=1,
    )

    matrix, columns = build_feature_matrix(items, smap, sid_canonical, amenities_top)

    # market_segment one-hot appended deterministically after the base plan.
    segment_vocab = _stable_vocabulary(items["market_segment"].astype(str))
    segment_columns = [f"market_segment={value}" for value in segment_vocab]
    segment_matrix = np.zeros((matrix.shape[0], len(segment_columns)), dtype=np.float32)
    canonical_to_segment = dict(
        zip(items["canonical_listing_id"], items["market_segment"].astype(str))
    )
    segment_index = {value: i for i, value in enumerate(segment_vocab)}
    for sid, dense_id in smap.items():
        canonical = sid_canonical.get(sid)
        segment = canonical_to_segment.get(canonical)
        position = segment_index.get(str(segment)) if segment is not None else None
        if position is not None:
            segment_matrix[dense_id, position] = 1.0
    matrix = np.concatenate((matrix, segment_matrix), axis=1)
    columns = columns + segment_columns

    _write_atomic_npy(matrix, output, force)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": int(matrix.shape[0]),
        "num_items": len(smap),
        "dimensions": int(matrix.shape[1]),
        "padding_row": 0,
        "columns": columns,
        "amenities_top": amenities_top,
        "alignment": "dense_id = dataset.smap[events.sid]; attributes via canonical_listing_id",
        "inputs": {
            "dataset_pkl_sha256": _sha256_file(dataset_path),
            "events_parquet_sha256": _sha256_tree(events_dir),
            "items_parquet_sha256": _sha256_tree(items_dir),
        },
        "transforms": {
            "log1p": list(NUMERIC_LOG_COLUMNS),
            "counts": list(NUMERIC_COUNT_COLUMNS),
            "categoricals": list(CATEGORICAL_COLUMNS),
            "coordinates": ["lat_region", "lon_region"],
            "missing_indicators": True,
            "fitted_statistics": False,
        },
        "matrix_sha256": _sha256_file(output),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def _market_segment(business_type: object, usage_type: object) -> str:
    business = str(business_type).strip().upper()
    usage = str(usage_type).strip().upper()
    if business in {"SALE", "SELL"}:
        business = "SALE"
    if usage == "RESIDENTIAL":
        return f"{business}_RESIDENTIAL"
    if usage == "COMMERCIAL":
        return f"{business}_COMMERCIAL"
    return "OTHER"


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "export-item-features",
        help="export a dense item-feature matrix aligned to dataset.pkl smap",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="trusted dataset.pkl")
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="directory with events.parquet and items.parquet",
    )
    parser.add_argument("--output", required=True, type=Path, help="target .npy path")
    parser.add_argument("--amenities-top", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> None:
    if args.amenities_top < 0:
        raise ValueError("amenities-top cannot be negative")
    dataset_path = args.dataset.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".npy":
        raise ValueError("output must end with .npy")
    manifest = export_item_features(
        dataset_path,
        source_dir,
        output,
        args.amenities_top,
        args.force,
    )
    print(f"Item features written to {output}")
    print(f"  rows: {manifest['rows']:_} (num_items={manifest['num_items']:_} + padding)")
    print(f"  dimensions: {manifest['dimensions']:_}")
    print(f"  manifest: {output}.manifest.json")
