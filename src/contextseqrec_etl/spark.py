from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


def make_spark(
    *,
    default_memory: str = "100g",
    default_local_dir: str = "/tmp/contextseqrec-etl",
) -> SparkSession:
    python_path = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
    local_dir = os.environ.get("SPARK_LOCAL_DIR", default_local_dir)
    cores = os.environ.get("SPARK_CORES", "16")
    memory = os.environ.get("SPARK_DRIVER_MEMORY", default_memory)
    shuffle_partitions = os.environ.get("SPARK_SHUFFLE_PARTITIONS", "400")

    return (
        SparkSession.builder.appName("contextseqrec-etl")
        .master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .config("spark.local.dir", local_dir)
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .config("spark.sql.autoBroadcastJoinThreshold", "268435456")
        .config("spark.shuffle.manager", "sort")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config(
            "spark.driver.extraJavaOptions",
            "-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+HeapDumpOnOutOfMemoryError",
        )
        .getOrCreate()
    )
