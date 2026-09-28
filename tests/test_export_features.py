from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contextseqrec_etl.export_features import (
    _parse_amenities,
    export_item_features,
    load_smap_sidecar,
    sid_to_canonical,
    write_smap_sidecar,
)


def _write_events(path: Path, rows: list[tuple[int, int, str]]) -> None:
    frame = pd.DataFrame(
        [
            {
                "uid": uid,
                "sid": sid,
                "canonical_listing_id": canonical,
                "event_type": "ListingRendered",
                "timestamp": index,
                "intensity": 1,
            }
            for index, (uid, sid, canonical) in enumerate(rows)
        ]
    )
    frame.to_parquet(path / "events.parquet")


def _write_items(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path / "items.parquet")


def _write_dataset(path: Path, smap: dict[int, int]) -> None:
    # Include heavy-looking keys to mimic the real pickle; only smap is read.
    payload = {
        "train": {1: [1, 2]},
        "val": {},
        "smap": smap,
        "umap": {10: 1},
        "bmap": {"ListingRendered": 2},
        "segmap": {"SALE_RESIDENTIAL": 1},
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    write_smap_sidecar(smap, path, force=False)


@pytest.fixture
def synthetic(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "work"
    source.mkdir()
    # Global sids 3,5,9 -> dense ids 1,2,3 (sorted order).
    _write_events(
        source,
        [
            (100, 3, "CANON_A"),
            (100, 5, "CANON_B"),
            (200, 9, "CANON_C"),
            (200, 5, "CANON_B"),
        ],
    )
    _write_items(
        source,
        [
            {
                "canonical_listing_id": "CANON_A",
                "listing_id_numeric": 111,
                "lat_region": "-20.006",
                "lon_region": "-43.96",
                "zip_code": "34007361",
                "neighborhood": "Serrana",
                "usable_areas": 150.0,
                "total_areas": 150.0,
                "bedrooms": 3,
                "bathrooms": 2,
                "suites": 1,
                "parking_spaces": 1,
                "price": 300000.0,
                "unit_type": "HOME",
                "amenities": "['GARAGE', 'POOL']",
                "business_type": "SALE",
                "usage_type": "RESIDENTIAL",
                "h3_res6": 1,
                "h3_res7": 2,
            },
            {
                "canonical_listing_id": "CANON_B",
                "listing_id_numeric": 222,
                "lat_region": "-18.9",
                "lon_region": "-48.2",
                "zip_code": "38411475",
                "neighborhood": "Centro",
                "usable_areas": 80.0,
                "total_areas": 90.0,
                "bedrooms": 2,
                "bathrooms": 1,
                "suites": None,
                "parking_spaces": None,
                "price": 1500.0,
                "unit_type": "APARTMENT",
                "amenities": "['GARAGE']",
                "business_type": "RENTAL",
                "usage_type": "RESIDENTIAL",
                "h3_res6": 3,
                "h3_res7": 4,
            },
            {
                "canonical_listing_id": "CANON_C",
                "listing_id_numeric": 333,
                "lat_region": "0",
                "lon_region": "0",
                "zip_code": "00000000",
                "neighborhood": "Unknown",
                "usable_areas": None,
                "total_areas": None,
                "bedrooms": 0,
                "bathrooms": 0,
                "suites": 0,
                "parking_spaces": 0,
                "price": None,
                "unit_type": "COMMERCIAL_ALLOTMENT_LAND",
                "amenities": "[]",
                "business_type": "SALE",
                "usage_type": "COMMERCIAL",
                "h3_res6": 5,
                "h3_res7": 6,
            },
        ],
    )
    dataset = tmp_path / "dataset.pkl"
    _write_dataset(dataset, {3: 1, 5: 2, 9: 3})
    return {"source": source, "dataset": dataset, "output": tmp_path / "item_features.npy"}


def test_parse_amenities_handles_edge_cases() -> None:
    assert _parse_amenities("['GARAGE', 'POOL']") == ["GARAGE", "POOL"]
    assert _parse_amenities("[]") == []
    assert _parse_amenities(None) == []
    assert _parse_amenities(float("nan")) == []
    assert _parse_amenities("garbage(") == []


def test_load_smap_sidecar_requires_contiguous_ids(tmp_path: Path) -> None:
    good = tmp_path / "good.pkl"
    _write_dataset(good, {7: 1, 8: 2, 9: 3})
    assert load_smap_sidecar(good) == {7: 1, 8: 2, 9: 3}
    bad = tmp_path / "bad.pkl"
    payload = {"smap": {7: 1, 8: 3}}
    with bad.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="contiguous"):
        write_smap_sidecar(payload["smap"], bad, force=False)


def test_sid_to_canonical_detects_conflicts(tmp_path: Path) -> None:
    source = tmp_path / "work"
    source.mkdir()
    _write_events(source, [(1, 5, "A"), (1, 5, "B")])
    with pytest.raises(ValueError, match="multiple canonical"):
        sid_to_canonical(source / "events.parquet")


def test_export_produces_aligned_matrix(synthetic: dict[str, Path]) -> None:
    manifest = export_item_features(
        synthetic["dataset"],
        synthetic["source"],
        synthetic["output"],
        amenities_top=64,
        force=False,
    )
    matrix = np.load(synthetic["output"], allow_pickle=False)
    columns = manifest["columns"]
    index = {name: i for i, name in enumerate(columns)}

    # Shape: M+1 rows (3 items + padding), finite, padding zeroed.
    assert matrix.shape[0] == 4
    assert manifest["num_items"] == 3
    assert np.isfinite(matrix).all()
    assert not matrix[0].any()

    # dense 1 = CANON_A: price present -> log1p; amenities GARAGE and POOL set.
    assert matrix[1, index["log1p_price"]] == pytest.approx(np.log1p(300000.0))
    assert matrix[1, index["missing_price"]] == 0.0
    assert matrix[1, index["amenity=GARAGE"]] == 1.0
    assert matrix[1, index["amenity=POOL"]] == 1.0
    assert matrix[1, index["market_segment=SALE_RESIDENTIAL"]] == 1.0

    # dense 2 = CANON_B: suites/parking missing -> indicators set.
    assert matrix[2, index["missing_suites"]] == 1.0
    assert matrix[2, index["missing_parking_spaces"]] == 1.0
    assert matrix[2, index["amenity=POOL"]] == 0.0

    # dense 3 = CANON_C: price/areas missing, coords zero -> missing indicators.
    assert matrix[3, index["missing_price"]] == 1.0
    assert matrix[3, index["missing_usable_areas"]] == 1.0
    assert matrix[3, index["missing_latitude"]] == 1.0
    assert matrix[3, index["missing_longitude"]] == 1.0
    assert matrix[3, index["market_segment=SALE_COMMERCIAL"]] == 1.0

    # Manifest integrity.
    assert manifest["schema_version"] == "contextseqrec-item-features-v2"
    assert manifest["dimensions"] == matrix.shape[1]
    assert len(columns) == matrix.shape[1]
    manifest_path = Path(str(synthetic["output"]) + ".manifest.json")
    stored = json.loads(manifest_path.read_text())
    assert stored["matrix_sha256"] == manifest["matrix_sha256"]


def test_export_refuses_overwrite_without_force(synthetic: dict[str, Path]) -> None:
    export_item_features(
        synthetic["dataset"], synthetic["source"], synthetic["output"], 64, force=False
    )
    with pytest.raises(FileExistsError):
        export_item_features(
            synthetic["dataset"], synthetic["source"], synthetic["output"], 64, force=False
        )
    # force=True replaces it.
    export_item_features(
        synthetic["dataset"], synthetic["source"], synthetic["output"], 64, force=True
    )


def test_matrix_matches_contextseqrec_loader_contract(synthetic: dict[str, Path]) -> None:
    manifest = export_item_features(
        synthetic["dataset"], synthetic["source"], synthetic["output"], 64, force=False
    )
    matrix = np.load(synthetic["output"], allow_pickle=False)
    num_items = manifest["num_items"]
    # ContextSecRec accepts M or M+1 rows; here we produce M+1 with zero padding.
    assert matrix.shape[0] in {num_items, num_items + 1}
    assert matrix.dtype == np.float32
    assert matrix.ndim == 2
    assert np.allclose(matrix[0], 0.0)



def test_export_rejects_model_item_missing_from_catalog(synthetic: dict[str, Path]) -> None:
    items = pd.read_parquet(synthetic["source"] / "items.parquet")
    items = items[items["canonical_listing_id"] != "CANON_C"]
    items.to_parquet(synthetic["source"] / "items.parquet")
    with pytest.raises(ValueError, match="absent from items.parquet"):
        export_item_features(
            synthetic["dataset"], synthetic["source"], synthetic["output"], 64, force=False
        )


def test_export_rejects_duplicate_catalog_ids(synthetic: dict[str, Path]) -> None:
    items = pd.read_parquet(synthetic["source"] / "items.parquet")
    pd.concat((items, items.iloc[[0]]), ignore_index=True).to_parquet(
        synthetic["source"] / "items.parquet"
    )
    with pytest.raises(ValueError, match="duplicate canonical"):
        export_item_features(
            synthetic["dataset"], synthetic["source"], synthetic["output"], 64, force=False
        )


def test_manifest_hashes_single_file_parquet_inputs(synthetic: dict[str, Path]) -> None:
    manifest = export_item_features(
        synthetic["dataset"], synthetic["source"], synthetic["output"], 64, force=False
    )
    import hashlib

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        value.update(path.read_bytes())
        return value.hexdigest()

    assert manifest["inputs"]["events_parquet_sha256"] == digest(
        synthetic["source"] / "events.parquet"
    )
    assert manifest["inputs"]["items_parquet_sha256"] == digest(
        synthetic["source"] / "items.parquet"
    )


def test_export_rejects_smap_sidecar_for_different_dataset(synthetic: dict[str, Path]) -> None:
    with synthetic["dataset"].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="different dataset"):
        export_item_features(
            synthetic["dataset"], synthetic["source"], synthetic["output"], 64, force=False
        )
