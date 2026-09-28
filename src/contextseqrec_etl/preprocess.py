from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import pandas as pd
from tqdm import tqdm

from contextseqrec_etl.export_features import write_smap_sidecar

CANONICAL_BEHAVIOR_IDS = {
    "RankingClicked": 1,
    "ListingRendered": 2,
    "FavoriteClicked": 3,
    "GalleryClicked": 4,
    "ShareClicked": 5,
    "DecisionTreeFormClicked": 6,
    "LeadPanelClicked": 7,
    "LeadClicked": 8,
}


def derive_market_segment(business_type: object, usage_type: object) -> str:
    business = str(business_type).strip().upper()
    usage = str(usage_type).strip().upper()
    if business in {"SALE", "SELL"}:
        business = "SALE"
    if usage == "RESIDENTIAL":
        return f"{business}_RESIDENTIAL"
    if usage == "COMMERCIAL":
        return f"{business}_COMMERCIAL"
    return "OTHER"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_frame(source_dir: Path) -> pd.DataFrame:
    events = pd.read_parquet(source_dir / "events.parquet")
    items = pd.read_parquet(source_dir / "items.parquet")
    _require_columns(
        events,
        {"uid", "sid", "canonical_listing_id", "event_type", "timestamp", "intensity"},
        "events.parquet",
    )
    _require_columns(
        items,
        {
            "canonical_listing_id",
            "business_type",
            "usage_type",
            "lat_region",
            "lon_region",
        },
        "items.parquet",
    )
    events = events.rename(columns={"event_type": "behavior"})
    items = items.drop_duplicates("canonical_listing_id").copy()
    items["market_segment"] = items.apply(
        lambda row: derive_market_segment(row["business_type"], row["usage_type"]),
        axis=1,
    )
    items["latitude"] = pd.to_numeric(items["lat_region"], errors="coerce")
    items["longitude"] = pd.to_numeric(items["lon_region"], errors="coerce")
    frame = events.merge(
        items[["canonical_listing_id", "latitude", "longitude", "market_segment"]],
        on="canonical_listing_id",
        how="left",
        validate="many_to_one",
    )
    frame["latitude"] = frame["latitude"].fillna(0.0)
    frame["longitude"] = frame["longitude"].fillna(0.0)
    frame["market_segment"] = frame["market_segment"].fillna("OTHER")
    frame["intensity"] = pd.to_numeric(frame["intensity"], errors="raise")
    if not frame["intensity"].map(lambda value: math.isfinite(float(value))).all():
        raise ValueError("intensity must contain only finite values")
    if (frame["intensity"] < 0).any():
        raise ValueError("intensity must be non-negative")
    return frame.sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)[
        [
            "uid",
            "sid",
            "behavior",
            "timestamp",
            "intensity",
            "market_segment",
            "latitude",
            "longitude",
        ]
    ]


def densify(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    umap = {value: index + 1 for index, value in enumerate(sorted(set(frame["uid"])))}
    smap = {value: index + 1 for index, value in enumerate(sorted(set(frame["sid"])))}
    present_behaviors = set(frame["behavior"])
    unknown = present_behaviors - set(CANONICAL_BEHAVIOR_IDS)
    if unknown:
        raise ValueError(f"unknown behaviors: {sorted(unknown)}")
    bmap = {
        behavior: behavior_id
        for behavior, behavior_id in CANONICAL_BEHAVIOR_IDS.items()
        if behavior in present_behaviors
    }
    segments = sorted(set(frame["market_segment"]))
    segmap = {segment: index + 1 for index, segment in enumerate(segments)}
    frame = frame.copy()
    frame["uid"] = frame["uid"].map(umap)
    frame["sid"] = frame["sid"].map(smap)
    frame["behavior"] = frame["behavior"].map(bmap)
    frame["market_segment"] = frame["market_segment"].map(segmap)
    return frame, umap, smap, bmap, segmap


def build_dataset(
    source_dir: Path,
    min_user_events: int,
    target_behavior: str,
) -> dict[str, Any]:
    frame = load_frame(source_dir)
    user_sizes = frame.groupby("uid").size()
    frame = frame[frame["uid"].isin(user_sizes.index[user_sizes >= min_user_events])].copy()
    if frame.empty:
        raise ValueError("no sessions remain after min-user-events filtering")
    frame, umap, smap, bmap, segmap = densify(frame)
    if target_behavior not in bmap:
        raise ValueError(f"target behavior is absent from the filtered data: {target_behavior}")
    target_id = bmap[target_behavior]

    train: dict[int, list[int]] = {}
    val: dict[int, list[int]] = {}
    train_b: dict[int, list[int]] = {}
    val_b: dict[int, list[int]] = {}
    train_intensity: dict[int, list] = {}
    val_intensity: dict[int, list] = {}
    train_seg: dict[int, list[int]] = {}
    val_seg: dict[int, list[int]] = {}
    train_lat: dict[int, list[float]] = {}
    val_lat: dict[int, list[float]] = {}
    train_lon: dict[int, list[float]] = {}
    val_lon: dict[int, list[float]] = {}

    groups = frame.groupby("uid", sort=True)
    for user, group in tqdm(groups, total=len(umap), desc="Splitting sessions"):
        user_id = int(user)
        items = [int(value) for value in group["sid"].tolist()]
        behaviors = [int(value) for value in group["behavior"].tolist()]
        intensity = group["intensity"].tolist()
        segments = [int(value) for value in group["market_segment"].tolist()]
        latitudes = [float(value) for value in group["latitude"].tolist()]
        longitudes = [float(value) for value in group["longitude"].tolist()]
        if behaviors[-1] == target_id:
            train[user_id], val[user_id] = items[:-1], items[-1:]
            train_b[user_id], val_b[user_id] = behaviors[:-1], behaviors[-1:]
            train_intensity[user_id], val_intensity[user_id] = intensity[:-1], intensity[-1:]
            train_seg[user_id], val_seg[user_id] = segments[:-1], segments[-1:]
            train_lat[user_id], val_lat[user_id] = latitudes[:-1], latitudes[-1:]
            train_lon[user_id], val_lon[user_id] = longitudes[:-1], longitudes[-1:]
        else:
            train[user_id] = items
            train_b[user_id] = behaviors
            train_intensity[user_id] = intensity
            train_seg[user_id] = segments
            train_lat[user_id] = latitudes
            train_lon[user_id] = longitudes

    return {
        "train": train,
        "val": val,
        "train_b": train_b,
        "val_b": val_b,
        "val_num": len(val),
        "umap": umap,
        "smap": smap,
        "bmap": bmap,
        "segmap": segmap,
        "train_intensity": train_intensity,
        "val_intensity": val_intensity,
        "train_seg": train_seg,
        "val_seg": val_seg,
        "train_lat": train_lat,
        "val_lat": val_lat,
        "train_lon": train_lon,
        "val_lon": val_lon,
    }


def write_dataset(dataset: dict[str, Any], output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"output already exists; pass --force to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary_name = stream.name
            pickle.dump(dataset, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "preprocess",
        help="convert final events/items Parquet into ContextSeqRec dataset.pkl",
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-user-events", type=int, default=5)
    parser.add_argument("--target-behavior", default="LeadClicked")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> None:
    if args.min_user_events < 2:
        raise ValueError("min-user-events must be at least 2")
    source_dir = args.source_dir.expanduser().resolve()
    if not (source_dir / "events.parquet").exists():
        raise FileNotFoundError(source_dir / "events.parquet")
    if not (source_dir / "items.parquet").exists():
        raise FileNotFoundError(source_dir / "items.parquet")
    output = args.output.expanduser().resolve()
    dataset = build_dataset(source_dir, args.min_user_events, args.target_behavior)
    write_dataset(dataset, output, args.force)
    smap_manifest = write_smap_sidecar(dataset["smap"], output, args.force)
    print(f"Dataset written atomically to {output}")
    print(f"  smap sidecar: {output}.smap.npz ({smap_manifest['rows']:_} rows)")
    print(f"  sessions: {len(dataset['umap']):_}")
    print(f"  items: {len(dataset['smap']):_}")
    print(f"  target holdouts: {len(dataset['val']):_}")
    print(f"  behavior map: {dataset['bmap']}")
