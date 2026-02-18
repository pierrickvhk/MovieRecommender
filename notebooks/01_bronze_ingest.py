# Databricks notebook source
DATA_ROOT = "dbfs:/Volumes/workspace/movielens/movielens_files"
print("Text line count:", spark.read.text(f"{DATA_ROOT}/movies.csv").count())


# COMMAND ----------

from pyspark.sql.functions import col

movies_debug_schema = """
movieId INT, title STRING, genres STRING, _corrupt_record STRING
"""

df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("mode", "PERMISSIVE")
    .option("columnNameOfCorruptRecord", "_corrupt_record")
    .option("quote", "\"")
    .option("escape", "\"")
    .schema(movies_debug_schema)
    .load(f"{DATA_ROOT}/movies.csv")
)

bad = df.filter(col("_corrupt_record").isNotNull())
print("Corrupt rows:", bad.count())
display(bad.limit(20))


# COMMAND ----------

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    IntegerType,
    DoubleType,
    StringType,
)

# ============================================================
# 01_bronze_ingest.py (Serverless-safe, robust CSV parsing)
# ============================================================

DATA_ROOT = "dbfs:/Volumes/workspace/movielens/movielens_files"
BRONZE_SCHEMA = "workspace.movielens_bronze"

# ---- Schemas (explicit contracts) ----
ratings_schema = StructType(
    [
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("rating", DoubleType(), False),
        StructField("timestamp", LongType(), True),
    ]
)

movies_schema = StructType(
    [
        StructField("movieId", IntegerType(), False),
        StructField("title", StringType(), False),
        StructField("genres", StringType(), True),
        # debug column (keep it in bronze; very useful to prove data quality)
        StructField("_corrupt_record", StringType(), True),
    ]
)

links_schema = StructType(
    [
        StructField("movieId", IntegerType(), False),
        StructField("imdbId", StringType(), True),
        StructField("tmdbId", StringType(), True),
    ]
)

tags_schema = StructType(
    [
        StructField("userId", IntegerType(), False),
        StructField("movieId", IntegerType(), False),
        StructField("tag", StringType(), True),
        StructField("timestamp", LongType(), True),
    ]
)

TABLES = {
    "bronze_ratings": ("ratings.csv", ratings_schema),
    "bronze_movies": ("movies.csv", movies_schema),
    "bronze_links": ("links.csv", links_schema),
    "bronze_tags": ("tags.csv", tags_schema),
}

def read_csv_strict(path: str, schema: StructType) -> DataFrame:
    # strict for files that are known clean
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("mode", "FAILFAST")
        .option("quote", "\"")
        .option("escape", "\"")
        .schema(schema)
        .load(path)
    )

def read_csv_permissive_with_corrupt(path: str, schema: StructType) -> DataFrame:
    # permissive for movies to avoid serverless failing on 1 bad row
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .option("quote", "\"")
        .option("escape", "\"")
        .schema(schema)
        .load(path)
    )

def ensure_schema(schema_name: str) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

def drop_tables(schema_name: str, table_names: list[str]) -> None:
    for t in table_names:
        spark.sql(f"DROP TABLE IF EXISTS {schema_name}.{t}")

def write_delta(df: DataFrame, full_table_name: str) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_table_name)
    )

def describe_table(full_table_name: str) -> None:
    display(spark.sql(f"DESCRIBE DETAIL {full_table_name}").select("format", "location"))

# -----------------------------
# 0) Preflight
# -----------------------------
print("=== Preflight: listing DATA_ROOT ===")
display(dbutils.fs.ls(DATA_ROOT))

# -----------------------------
# 1) Create schema + clean slate
# -----------------------------
print("\n=== Create schema + drop existing bronze tables ===")
ensure_schema(BRONZE_SCHEMA)
drop_tables(BRONZE_SCHEMA, list(TABLES.keys()))

# -----------------------------
# 2) Read -> Write
# -----------------------------
print("\n=== Read -> Write Bronze Delta tables ===")

for table, (fname, schema) in TABLES.items():
    src = f"{DATA_ROOT}/{fname}"
    full = f"{BRONZE_SCHEMA}.{table}"
    print(f"\n-- Ingesting {src} -> {full}")

    if table == "bronze_movies":
        df = read_csv_permissive_with_corrupt(src, schema)

        # quality insight
        bad = df.filter(col("_corrupt_record").isNotNull())
        bad_count = bad.count()
        print(f"Corrupt rows in movies.csv: {bad_count}")
        if bad_count > 0:
            display(bad.limit(20))
            # Optional: hard fail (uncomment if you want pipeline to stop)
            # raise RuntimeError(f"movies.csv has {bad_count} corrupt rows. Fix CSV or relax parsing.")

        print("Preview movies:")
        display(df.select("movieId", "title", "genres").limit(5))
    else:
        df = read_csv_strict(src, schema)

    write_delta(df, full)
    describe_table(full)

# -----------------------------
# 3) Verify counts
# -----------------------------
print("\n=== Verify counts from Delta tables ===")
for t in TABLES.keys():
    full = f"{BRONZE_SCHEMA}.{t}"
    c = spark.table(full).count()
    print(f"{full}: {c}")

print("\nDONE: Bronze ingest stored as Delta.")
