from __future__ import annotations

from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F


def prepare_sessions(df: DataFrame) -> DataFrame:
    """Collapse consecutive item/behavior bursts and count their intensity."""
    print("  Preparing sessions (burst collapse by item and behavior)...")
    df = (
        df.withColumn("user_id", F.col("anonymized_session_id"))
        .withColumn("item_id", F.col("canonical_listing_id"))
        .withColumn("timestamp", F.unix_timestamp(F.col("event_ts")))
        .filter(
            F.col("user_id").isNotNull()
            & F.col("item_id").isNotNull()
            & F.col("timestamp").isNotNull()
        )
        .withColumn("tie_breaker", F.monotonically_increasing_id())
    )
    order_spec = Window.partitionBy("user_id").orderBy("timestamp", "tie_breaker")
    df = df.withColumn("prev_item", F.lag("item_id").over(order_spec)).withColumn(
        "prev_behavior", F.lag("event_type").over(order_spec)
    )
    is_new_run = (
        F.col("prev_item").isNull()
        | (F.col("item_id") != F.col("prev_item"))
        | (F.col("event_type") != F.col("prev_behavior"))
    )
    run_window = order_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df = df.withColumn("is_new_run", F.when(is_new_run, 1).otherwise(0)).withColumn(
        "run_seq", F.sum("is_new_run").over(run_window)
    )
    collapsed = df.groupBy("user_id", "run_seq").agg(
        F.first("item_id", ignorenulls=True).alias("item_id"),
        F.first("event_type", ignorenulls=True).alias("event_type"),
        F.min("timestamp").alias("timestamp"),
        F.count(F.lit(1)).alias("intensity"),
    )
    print(f"    After burst collapse: {collapsed.count():_} interactions")
    return collapsed


def filter_sessions_by_length(
    df: DataFrame,
    min_length: int = 2,
    max_length: int = 50,
) -> DataFrame:
    """Discard short sessions and retain the most recent events in long sessions."""
    print(f"  Filtering sessions ({min_length}-{max_length})...")
    session_sizes = df.groupBy("user_id").agg(F.count("*").alias("session_size"))
    df = df.join(session_sizes, on="user_id", how="inner").filter(
        F.col("session_size") >= min_length
    )
    recent_first = Window.partitionBy("user_id").orderBy(F.col("timestamp").desc())
    df = (
        df.withColumn("row_number", F.row_number().over(recent_first))
        .filter((F.col("session_size") <= max_length) | (F.col("row_number") <= max_length))
        .drop("row_number", "session_size")
        .orderBy("user_id", "timestamp")
    )
    print(f"    Sessions remaining: {df.select('user_id').distinct().count():_}")
    return df


def filter_rare_items(df: DataFrame, min_support: int = 2) -> DataFrame:
    """Discard items with fewer than min_support collapsed interactions."""
    print(f"  Filtering rare items (minimum {min_support})...")
    valid_items = (
        df.groupBy("item_id")
        .agg(F.count("*").alias("item_count"))
        .filter(F.col("item_count") >= min_support)
        .select("item_id")
    )
    df = df.join(valid_items, on="item_id", how="inner")
    print(f"    Items remaining: {df.select('item_id').distinct().count():_}")
    return df


def create_numeric_ids(df: DataFrame) -> DataFrame:
    """Assign deterministic dense positive IDs to sessions and items."""
    print("  Creating numeric IDs...")
    session_map = (
        df.select("user_id")
        .distinct()
        .withColumn(
            "session_id_numeric",
            F.row_number().over(Window.orderBy("user_id")),
        )
    )
    df = df.join(session_map, "user_id", "inner")
    item_map = (
        df.select("item_id")
        .distinct()
        .withColumn(
            "item_id_numeric",
            F.row_number().over(Window.orderBy("item_id")),
        )
    )
    df = df.join(item_map, "item_id", "inner")
    print(f"    Numeric IDs: {session_map.count():_} sessions, {item_map.count():_} items")
    return df


def run_session_pipeline(
    df: DataFrame,
    min_session_length: int = 2,
    max_session_length: int = 50,
    min_item_freq: int = 2,
) -> DataFrame:
    """Run burst collapse, filtering, refiltering and deterministic densification."""
    print("Session pipeline: running...")
    df = prepare_sessions(df)
    df = filter_sessions_by_length(df, min_session_length, max_session_length)
    df = filter_rare_items(df, min_item_freq)
    df = filter_sessions_by_length(df, min_session_length, max_session_length)
    df = create_numeric_ids(df)
    total = df.count()
    sessions = df.select("user_id").distinct().count()
    items = df.select("item_id").distinct().count()
    print(f"  Final: {total:_} interactions, {sessions:_} sessions, {items:_} items")
    return df
