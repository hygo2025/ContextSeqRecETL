from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
import pyspark.sql.functions as F

from contextseqrec_etl.sessions import run_session_pipeline
from contextseqrec_etl.spark import make_spark

INTERACTION_TYPES = (
    "ListingRendered",
    "GalleryClicked",
    "RankingClicked",
    "LeadPanelClicked",
    "LeadClicked",
    "FavoriteClicked",
    "ShareClicked",
    "DecisionTreeFormClicked",
)


def _require_columns(df: DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def clean_listing_data(df: DataFrame) -> DataFrame:
    for column in ("price", "usable_areas", "total_areas", "ceiling_height"):
        if column in df.columns:
            df = df.withColumn(
                column,
                F.regexp_replace(F.col(column), r"[^0-9.]", "").cast("double"),
            )
    for column in ("bathrooms", "bedrooms", "suites", "parking_spaces", "floors"):
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast("integer"))
    if "dt" in df.columns:
        df = df.withColumn("dt", F.to_date(F.col("dt")))
    if "created_at" in df.columns:
        df = df.withColumn("created_at", F.to_timestamp(F.col("created_at")))
    if "updated_at" in df.columns:
        df = df.withColumn("updated_at", F.to_timestamp(F.col("updated_at")))
    return df


def add_h3_cells(
    df: DataFrame,
    resolutions: list[int],
    lat_column: str = "lat_region",
    lon_column: str = "lon_region",
) -> DataFrame:
    from pyspark.sql.types import LongType

    def make_cell(resolution: int):
        def cell(latitude: object, longitude: object) -> int:
            import h3

            try:
                lat = float(latitude)
                lon = float(longitude)
            except (TypeError, ValueError):
                return 0
            if lat == 0.0 and lon == 0.0:
                return 0
            return int(h3.str_to_int(h3.latlng_to_cell(lat, lon, resolution)))

        return F.udf(cell, LongType())

    for resolution in resolutions:
        df = df.withColumn(
            f"h3_res{resolution}",
            make_cell(resolution)(F.col(lat_column), F.col(lon_column)),
        )
    return df


def deduplicate_and_map_listings(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    active = df.filter(F.col("status") == "ACTIVE")
    latest = (
        active.withColumn(
            "row_number",
            F.row_number().over(
                Window.partitionBy("anonymized_listing_id").orderBy(F.col("updated_at").desc())
            ),
        )
        .filter(F.col("row_number") == 1)
        .drop("row_number")
        .withColumn("canonical_listing_id", F.col("anonymized_listing_id"))
    )
    canonical_to_numeric = (
        latest.select("canonical_listing_id")
        .distinct()
        .withColumn(
            "listing_id_numeric",
            F.row_number().over(Window.orderBy("canonical_listing_id")),
        )
    )
    mapping = (
        latest.select("anonymized_listing_id", "canonical_listing_id")
        .distinct()
        .join(canonical_to_numeric, "canonical_listing_id", "inner")
    )
    latest = latest.join(
        mapping.select("anonymized_listing_id", "listing_id_numeric"),
        "anonymized_listing_id",
        "inner",
    )
    return latest, mapping


def run_listings_pipeline(
    spark: SparkSession,
    raw_paths: list[str],
    output_path: Path,
    h3_resolutions: list[int],
) -> None:
    print("Listings pipeline: loading...")
    raw = spark.read.option("header", "true").csv(raw_paths)
    _require_columns(
        raw,
        {
            "anonymized_listing_id",
            "status",
            "updated_at",
            "lat_region",
            "lon_region",
            "business_type",
            "usage_type",
        },
        "listings input",
    )
    print(f"  Raw listings: {raw.count():_}")
    listings, mapping = deduplicate_and_map_listings(clean_listing_data(raw))
    listings = add_h3_cells(listings, h3_resolutions)
    print(f"  Active deduplicated: {listings.count():_}")
    print(f"  Mapping entries: {mapping.count():_}")
    listings.write.mode("overwrite").parquet(str(output_path / "listings_processed"))
    mapping.write.mode("overwrite").parquet(str(output_path / "listing_id_mapping"))

    preferred_columns = [
        "canonical_listing_id",
        "listing_id_numeric",
        "lat_region",
        "lon_region",
        "zip_code",
        "neighborhood",
        "usable_areas",
        "total_areas",
        "bedrooms",
        "bathrooms",
        "suites",
        "parking_spaces",
        "price",
        "unit_type",
        "amenities",
        "business_type",
        "usage_type",
    ]
    preferred_columns.extend(column for column in listings.columns if column.startswith("h3_res"))
    available = [column for column in preferred_columns if column in listings.columns]
    (
        listings.select(available)
        .dropDuplicates(["canonical_listing_id"])
        .coalesce(1)
        .write.mode("overwrite")
        .parquet(str(output_path / "items.parquet"))
    )
    print("Listings pipeline done.")


def clean_event_data(df: DataFrame) -> DataFrame:
    for column in (
        "anonymized_user_id",
        "anonymized_anonymous_id",
        "anonymized_listing_id",
        "anonymized_session_id",
    ):
        if column in df.columns:
            df = df.withColumn(
                column,
                F.when(F.trim(F.col(column)) == "", None).otherwise(F.trim(F.col(column))),
            )
    return df.withColumn(
        "event_ts",
        (F.col("collector_timestamp").cast("bigint") / 1000).cast("timestamp"),
    )


def run_events_pipeline(
    spark: SparkSession,
    raw_paths: list[str],
    mapping_path: Path,
    output_path: Path,
) -> None:
    print("Events pipeline: loading...")
    raw = spark.read.option("header", "true").csv(raw_paths)
    _require_columns(
        raw,
        {
            "anonymized_listing_id",
            "anonymized_session_id",
            "collector_timestamp",
            "event_type",
        },
        "events input",
    )
    mapping = spark.read.parquet(str(mapping_path))
    events = clean_event_data(
        raw.join(F.broadcast(mapping), on="anonymized_listing_id", how="inner")
    ).filter(F.col("event_type").isin(*INTERACTION_TYPES))
    events.write.mode("overwrite").parquet(str(output_path / "events_processed"))
    print("Events pipeline done.")


def step_done(path: Path) -> bool:
    return (path / "_SUCCESS").exists()


def _source_subdir(data_root: Path, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if (data_root / candidate).is_dir():
            return candidate
    raise FileNotFoundError(
        f"none of the expected source directories exist under {data_root}: {candidates}"
    )


def _raw_pattern(data_root: Path, subdir: str, region: str) -> str:
    directory = data_root / subdir / region
    files = list(directory.glob("*.csv.gz")) if directory.is_dir() else []
    if not files:
        raise FileNotFoundError(f"no .csv.gz files found in {directory}")
    return str(directory / "*.csv.gz")


def process_region(
    spark: SparkSession,
    data_root: Path,
    out_dir: Path,
    region: str,
    args: argparse.Namespace,
) -> None:
    region_dir = out_dir / region
    region_dir.mkdir(parents=True, exist_ok=True)
    listings_subdir = _source_subdir(data_root, ("listings", "listing"))
    events_subdir = _source_subdir(data_root, ("events", "event"))
    listing_paths = [_raw_pattern(data_root, listings_subdir, region)]
    event_paths = [_raw_pattern(data_root, events_subdir, region)]

    print("=" * 60)
    print(f"Region: {region}")
    print("=" * 60)
    listings_out = region_dir / "listings_processed"
    mapping_out = region_dir / "listing_id_mapping"
    if (
        step_done(listings_out)
        and step_done(mapping_out)
        and step_done(region_dir / "items.parquet")
    ):
        print(f"  [SKIP] Listings already complete for {region}")
    else:
        run_listings_pipeline(spark, listing_paths, region_dir, args.h3_res)

    events_out = region_dir / "events_processed"
    if step_done(events_out):
        print(f"  [SKIP] Events already complete for {region}")
    else:
        run_events_pipeline(spark, event_paths, mapping_out, region_dir)

    sessions_out = region_dir / "sessions_processed"
    if step_done(sessions_out):
        print(f"  [SKIP] Sessions already complete for {region}")
    else:
        sessions = run_session_pipeline(
            spark.read.parquet(str(events_out)),
            min_session_length=args.min_session_length,
            max_session_length=args.max_session_length,
            min_item_freq=args.min_item_freq,
        )
        sessions.select(
            sessions["session_id_numeric"].alias("uid"),
            sessions["item_id_numeric"].alias("sid"),
            sessions["item_id"].alias("canonical_listing_id"),
            "event_type",
            "timestamp",
            "intensity",
        ).write.mode("overwrite").parquet(str(sessions_out))
        print(f"  Sessions written to {sessions_out}")


def write_final_output(
    spark: SparkSession,
    out_dir: Path,
    regions: list[str],
    output_partitions: int,
) -> None:
    print("=" * 60)
    print("Final merge: combining regions")
    print("=" * 60)
    region_dirs = [out_dir / region for region in regions]
    incomplete = [
        region.name
        for region in region_dirs
        if not step_done(region / "sessions_processed") or not step_done(region / "items.parquet")
    ]
    if incomplete:
        raise RuntimeError(f"regions without complete session/item outputs: {incomplete}")

    parts: list[DataFrame] = []
    uid_offset = 0
    sid_offset = 0
    for region in region_dirs:
        part = spark.read.parquet(str(region / "sessions_processed"))
        if uid_offset:
            part = part.withColumn("uid", F.col("uid") + uid_offset)
        if sid_offset:
            part = part.withColumn("sid", F.col("sid") + sid_offset)
        max_uid, max_sid = part.agg(F.max("uid"), F.max("sid")).first()
        if max_uid is None or max_sid is None:
            raise RuntimeError(f"region {region.name} produced no sessions")
        uid_offset = int(max_uid)
        sid_offset = int(max_sid)
        parts.append(part)
        print(f"  {region.name}: cumulative uid={uid_offset}, sid={sid_offset}")

    final = reduce(lambda left, right: left.unionByName(right), parts)
    events_out = out_dir / "events.parquet"
    final.coalesce(output_partitions).write.mode("overwrite").parquet(str(events_out))
    items = (
        spark.read.parquet(*(str(region / "items.parquet") for region in region_dirs))
        .dropDuplicates(["canonical_listing_id"])
        .coalesce(1)
    )
    items.write.mode("overwrite").parquet(str(out_dir / "items.parquet"))
    print(
        f"Done: {events_out} contains {final.count():_} rows, "
        f"{final.select('uid').distinct().count():_} sessions and "
        f"{final.select('sid').distinct().count():_} item IDs"
    )


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "convert",
        help="convert regional compressed CSV files into events/items Parquet",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--regions", nargs="+", default=["mg", "es_bh"])
    parser.add_argument("--min-session-length", type=int, default=3)
    parser.add_argument("--max-session-length", type=int, default=50)
    parser.add_argument("--min-item-freq", type=int, default=5)
    parser.add_argument("--h3-res", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--output-partitions", type=int, default=200)
    parser.add_argument("--final-only", action="store_true")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> None:
    if args.min_session_length < 2:
        raise ValueError("min-session-length must be at least 2")
    if args.max_session_length < args.min_session_length:
        raise ValueError("max-session-length must be at least min-session-length")
    if args.min_item_freq <= 0 or args.output_partitions <= 0:
        raise ValueError("min-item-freq and output-partitions must be positive")
    if not args.regions or len(set(args.regions)) != len(args.regions):
        raise ValueError("regions must be non-empty and unique")

    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    spark = make_spark()
    try:
        if not args.final_only:
            for region in args.regions:
                process_region(spark, data_root, out_dir, region, args)
        write_final_output(spark, out_dir, args.regions, args.output_partitions)
    finally:
        spark.stop()
