from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    IntegerType,
    LongType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Schemas:
    ratings: StructType
    movies: StructType
    links: StructType
    scraped: StructType


def default_schemas() -> Schemas:
    ratings = StructType(
        [
            StructField("userId", IntegerType(), False),
            StructField("movieId", IntegerType(), False),
            StructField("rating", DoubleType(), False),
            StructField("timestamp", LongType(), True),
        ]
    )

    movies = StructType(
        [
            StructField("movieId", IntegerType(), False),
            StructField("title", StringType(), False),
            StructField("genres", StringType(), True),
        ]
    )

    links = StructType(
        [
            StructField("movieId", IntegerType(), False),
            StructField("imdbId", StringType(), True),
            StructField("tmdbId", StringType(), True),
        ]
    )

    # Houd scraped flexibel maar toch met contract
    scraped = StructType(
        [
            StructField("movieId", IntegerType(), True),
            StructField("imdbId", StringType(), True),
            StructField("tmdbId", StringType(), True),
            StructField("title", StringType(), True),
            StructField("director", StringType(), True),
            StructField("budget", StringType(), True),
            StructField("poster_url", StringType(), True),
            StructField("source", StringType(), True),
            StructField("ingested_at", StringType(), True),
        ]
    )

    return Schemas(ratings=ratings, movies=movies, links=links, scraped=scraped)


def read_csv(
    spark: SparkSession,
    path: str,
    schema: StructType,
    header: bool = True,
) -> DataFrame:
    logger.info("Reading CSV: %s", path)
    return (
        spark.read.format("csv")
        .option("header", str(header).lower())
        .option("mode", "FAILFAST")
        .schema(schema)
        .load(path)
    )


def read_json(
    spark: SparkSession,
    path: str,
    schema: Optional[StructType] = None,
) -> DataFrame:
    logger.info("Reading JSON: %s", path)
    reader = spark.read.format("json").option("mode", "PERMISSIVE")
    if schema is not None:
        reader = reader.schema(schema)
    return reader.load(path)


def write_delta_table(df: DataFrame, table_name: str, mode: str = "overwrite") -> None:
    logger.info("Writing Delta table %s (mode=%s)", table_name, mode)
    (
        df.write.format("delta")
        .mode(mode)
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
