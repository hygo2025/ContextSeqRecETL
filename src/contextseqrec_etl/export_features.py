from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "contextseqrec-item-features-v2"
SMAP_SCHEMA_VERSION = "contextseqrec-smap-v1"

LOG_COLUMNS = ("price", "usable_areas", "total_areas")
COUNT_COLUMNS = ("bedrooms", "bathrooms", "suites", "parking_spaces")
ONE_HOT_COLUMNS = ("business_type", "usage_type", "unit_type", "market_segment")
HASH_COLUMNS = {"zip_code": 32, "neighborhood": 64, "h3_res6": 32, "h3_res7": 64}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_path(path: Path) -> str:
    return _sha256_tree(path) if path.is_dir() else _sha256_file(path)


def _atomic_write_bytes(payload: bytes, output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"output already exists; pass --force to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _atomic_write_npy(matrix: np.ndarray, output: Path, force: bool) -> None:
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


def smap_sidecar_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(dataset_path.suffix + ".smap.npz")


def smap_manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(dataset_path.suffix + ".smap.manifest.json")


def write_smap_sidecar(smap: dict[Any, Any], dataset_path: Path, force: bool) -> dict[str, Any]:
    dense = {int(sid): int(dense_id) for sid, dense_id in smap.items()}
    dense_ids = sorted(dense.values())
    if dense_ids != list(range(1, len(dense_ids) + 1)):
        raise ValueError("smap dense ids must be a contiguous range starting at 1")
    ordered = sorted(dense.items(), key=lambda pair: pair[1])
    sid = np.asarray([value for value, _ in ordered], dtype=np.int32)
    dense_id = np.asarray([value for _, value in ordered], dtype=np.int32)
    output = smap_sidecar_path(dataset_path)
    if output.exists() and not force:
        raise FileExistsError(f"output already exists; pass --force to replace it: {output}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False, suffix=".npz") as stream:
            temporary_name = stream.name
            np.savez_compressed(stream, sid=sid, dense_id=dense_id)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    manifest = {
        "schema_version": SMAP_SCHEMA_VERSION,
        "dataset_pkl_sha256": _sha256_file(dataset_path),
        "rows": int(sid.shape[0]),
        "sid_dtype": str(sid.dtype),
        "dense_id_dtype": str(dense_id.dtype),
        "sidecar_sha256": _sha256_file(output),
    }
    _atomic_write_bytes(
        json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        smap_manifest_path(dataset_path),
        force,
    )
    return manifest


def load_smap_sidecar(dataset_path: Path) -> dict[int, int]:
    sidecar = smap_sidecar_path(dataset_path)
    manifest_path = smap_manifest_path(dataset_path)
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"smap sidecar not found: {sidecar}; rerun preprocess with the current ETL"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"smap manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SMAP_SCHEMA_VERSION:
        raise ValueError("unsupported smap sidecar schema")
    if manifest.get("dataset_pkl_sha256") != _sha256_file(dataset_path):
        raise ValueError("smap sidecar belongs to a different dataset.pkl")
    if manifest.get("sidecar_sha256") != _sha256_file(sidecar):
        raise ValueError("smap sidecar hash does not match its manifest")
    archive = np.load(sidecar, allow_pickle=False)
    try:
        if set(archive.files) != {"sid", "dense_id"}:
            raise ValueError("smap sidecar must contain exactly sid and dense_id arrays")
        sid = np.asarray(archive["sid"], dtype=np.int64)
        dense_id = np.asarray(archive["dense_id"], dtype=np.int64)
    finally:
        archive.close()
    if sid.ndim != 1 or dense_id.ndim != 1 or sid.shape != dense_id.shape or not len(sid):
        raise ValueError("smap sidecar arrays must be non-empty aligned vectors")
    if len(set(sid.tolist())) != len(sid):
        raise ValueError("smap sidecar contains duplicate sid values")
    if sorted(dense_id.tolist()) != list(range(1, len(dense_id) + 1)):
        raise ValueError("smap sidecar dense ids must be contiguous starting at 1")
    return {int(source): int(target) for source, target in zip(sid, dense_id)}


def sid_to_canonical(events_path: Path) -> dict[int, str]:
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(events_path), format="parquet")
    mapping: dict[int, str] = {}
    scanner = dataset.scanner(columns=["sid", "canonical_listing_id"], batch_size=1_000_000)
    for batch in scanner.to_batches():
        for sid, canonical in zip(
            batch.column("sid").to_pylist(), batch.column("canonical_listing_id").to_pylist()
        ):
            if sid is None or canonical is None:
                raise ValueError("events.parquet contains null sid or canonical_listing_id")
            source_id = int(sid)
            previous = mapping.setdefault(source_id, str(canonical))
            if previous != canonical:
                raise ValueError(
                    f"sid {source_id} maps to multiple canonical ids: {previous!r} and {canonical!r}"
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
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(item).strip().upper() for item in parsed if str(item).strip()]


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


def _stable_vocabulary(values: pd.Series, limit: int | None = None) -> list[str]:
    counts = values.value_counts()
    ordered = sorted(counts.index, key=lambda value: (-int(counts[value]), str(value)))
    result = [str(value) for value in ordered]
    return result if limit is None else result[:limit]


def _hash_bucket(value: object, buckets: int) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % buckets


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_catalog(items: pd.DataFrame) -> pd.DataFrame:
    required = {"canonical_listing_id", "business_type", "usage_type"}
    missing = sorted(required - set(items.columns))
    if missing:
        raise ValueError(f"items.parquet is missing required columns: {missing}")
    duplicate = items[items.duplicated("canonical_listing_id", keep=False)][
        "canonical_listing_id"
    ]
    if not duplicate.empty:
        examples = sorted(set(duplicate.astype(str)))[:5]
        raise ValueError(f"items.parquet has duplicate canonical ids, e.g. {examples}")
    result = items.copy()
    result["market_segment"] = [
        _market_segment(business, usage)
        for business, usage in zip(result.get("business_type"), result.get("usage_type"))
    ]
    return result.set_index("canonical_listing_id", verify_integrity=True)


def build_feature_matrix(
    items: pd.DataFrame,
    smap: dict[int, int],
    sid_canonical: dict[int, str],
    amenities_top: int,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    catalog = _validate_catalog(items)
    num_items = len(smap)
    dense_to_canonical: list[str | None] = [None] * (num_items + 1)
    missing_event_sids: list[int] = []
    for sid, dense_id in smap.items():
        canonical = sid_canonical.get(sid)
        if canonical is None:
            missing_event_sids.append(sid)
        else:
            dense_to_canonical[dense_id] = canonical
    if missing_event_sids:
        raise ValueError(
            f"{len(missing_event_sids)} smap sids are absent from events.parquet; "
            f"examples={missing_event_sids[:5]}"
        )
    canonicals = pd.Index(dense_to_canonical[1:])
    missing_catalog = canonicals.difference(catalog.index)
    if not missing_catalog.empty:
        raise ValueError(
            f"{len(missing_catalog)} model items are absent from items.parquet; "
            f"examples={missing_catalog[:5].tolist()}"
        )
    aligned = catalog.reindex(canonicals)

    columns: list[str] = []
    for name in LOG_COLUMNS:
        columns.extend((f"log1p_{name}", f"missing_{name}"))
    for name in COUNT_COLUMNS:
        columns.extend((name, f"missing_{name}"))
    columns.extend(("latitude", "longitude", "missing_latitude", "missing_longitude"))

    categorical_vocab: dict[str, list[str]] = {}
    for name in ONE_HOT_COLUMNS:
        if name in aligned.columns:
            categorical_vocab[name] = _stable_vocabulary(aligned[name].dropna().astype(str))
            columns.extend(f"{name}={value}" for value in categorical_vocab[name])

    amenity_vocab: list[str] = []
    if "amenities" in aligned.columns and amenities_top:
        parsed_amenities = aligned["amenities"].map(_parse_amenities)
        flat = pd.Series([value for row in parsed_amenities for value in row], dtype="object")
        amenity_vocab = _stable_vocabulary(flat, amenities_top) if not flat.empty else []
        columns.extend(f"amenity={value}" for value in amenity_vocab)
    else:
        parsed_amenities = pd.Series([[]] * len(aligned), index=aligned.index)

    for name, buckets in HASH_COLUMNS.items():
        columns.append(f"missing_{name}")
        columns.extend(f"{name}_hash_{index}" for index in range(buckets))

    index = {name: position for position, name in enumerate(columns)}
    matrix = np.zeros((num_items + 1, len(columns)), dtype=np.float32)

    for offset, (_, row) in enumerate(aligned.iterrows(), start=1):
        for name in LOG_COLUMNS:
            number = _to_float(row.get(name)) if name in aligned.columns else None
            if number is None or number < 0:
                matrix[offset, index[f"missing_{name}"]] = 1.0
            else:
                matrix[offset, index[f"log1p_{name}"]] = math.log1p(number)
        for name in COUNT_COLUMNS:
            number = _to_float(row.get(name)) if name in aligned.columns else None
            if number is None:
                matrix[offset, index[f"missing_{name}"]] = 1.0
            else:
                matrix[offset, index[name]] = number

        for column, feature in (("lat_region", "latitude"), ("lon_region", "longitude")):
            number = _to_float(row.get(column)) if column in aligned.columns else None
            if number is None or number == 0.0:
                matrix[offset, index[f"missing_{feature}"]] = 1.0
            else:
                matrix[offset, index[feature]] = number

        for name, vocab in categorical_vocab.items():
            value = row.get(name)
            key = f"{name}={value}"
            if value is not None and key in index:
                matrix[offset, index[key]] = 1.0

        for amenity in parsed_amenities.iloc[offset - 1]:
            position = index.get(f"amenity={amenity}")
            if position is not None:
                matrix[offset, position] = 1.0

        for name, buckets in HASH_COLUMNS.items():
            bucket = _hash_bucket(row.get(name), buckets) if name in aligned.columns else None
            if bucket is None:
                matrix[offset, index[f"missing_{name}"]] = 1.0
            else:
                matrix[offset, index[f"{name}_hash_{bucket}"]] = 1.0

    if not np.isfinite(matrix).all() or matrix[0].any():
        raise ValueError("feature matrix violates finite-value or padding invariants")
    return matrix, columns, {"catalog_items": len(aligned), "missing_catalog_items": 0}


def export_item_features(
    dataset_path: Path,
    source_dir: Path,
    output: Path,
    amenities_top: int,
    force: bool,
) -> dict[str, Any]:
    events_path = source_dir / "events.parquet"
    items_path = source_dir / "items.parquet"
    if not events_path.exists() or not items_path.exists():
        raise FileNotFoundError("source-dir must contain events.parquet and items.parquet")
    smap = load_smap_sidecar(dataset_path)
    sid_canonical = sid_to_canonical(events_path)
    items = pd.read_parquet(items_path)
    matrix, columns, coverage = build_feature_matrix(items, smap, sid_canonical, amenities_top)

    _atomic_write_npy(matrix, output, force)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": int(matrix.shape[0]),
        "num_items": len(smap),
        "dimensions": int(matrix.shape[1]),
        "padding_row": 0,
        "columns": columns,
        "amenities_top": amenities_top,
        "alignment": "dense_id = dataset.smap[events.sid]; attributes via canonical_listing_id",
        "metadata_regime": "catalog_snapshot_transductive",
        "coverage": coverage,
        "inputs": {
            "dataset_pkl_sha256": _sha256_file(dataset_path),
            "smap_sidecar_sha256": _hash_path(smap_sidecar_path(dataset_path)),
            "events_parquet_sha256": _hash_path(events_path),
            "items_parquet_sha256": _hash_path(items_path),
        },
        "transforms": {
            "log1p": list(LOG_COLUMNS),
            "counts": list(COUNT_COLUMNS),
            "one_hot": list(ONE_HOT_COLUMNS),
            "hashed": HASH_COLUMNS,
            "missing_indicators": True,
            "fitted_statistics": False,
            "data_derived_vocabularies": True,
        },
        "matrix_sha256": _sha256_file(output),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _atomic_write_bytes(
        json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"), manifest_path, force
    )
    return manifest


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "export-item-features", help="export dense item features aligned to a dataset smap"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
