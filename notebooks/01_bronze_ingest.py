# Databricks notebook source
from __future__ import annotations

import logging

from pyspark.sql import functions as F

from movielens.config import BronzeTables, Paths
from movielens.io import default_schemas, read_csv, read_json, write_delta_table

logging.basicConfig(level=logging.INFO)

# --- Widgets ---
dbutils.widgets.text("data_root", "dbfs:/FileStore/movielens")  # pas aan
data_root = dbutils.widgets.get("data_root")

paths = Paths(data_root=data_root)
tables = BronzeTables()
schemas = default_schemas()

# --- Read sources ---
ratings_df = read_csv(spark, paths.ratings_path(), schemas.ratings)
movies_df = read_csv(spark, paths.movies_path(), schemas.movies)
links_df = read_csv(spark, paths.links_path(), schemas.links)

# Scraped
scraped_raw = read_json(spark, paths.scraped_path(), schemas.scraped)

# Minimal hygiene: add ingested_at if missing (Bronze)
if "ingested_at" not in scraped_raw.columns:
    scraped_df = scraped_raw.withColumn("ingested_at", F.current_timestamp().cast("string"))
else:
    scraped_df = scraped_raw

# --- Write Bronze tables (idempotent) ---
write_delta_table(ratings_df, tables.ratings)
write_delta_table(movies_df, tables.movies)
write_delta_table(links_df, tables.links)
write_delta_table(scraped_df, tables.scraped)


# --- Run summary ---
def summarize(name: str) -> None:
    df = spark.table(name)
    print(f"\n=== {name} ===")
    print("rows:", df.count())
    df.printSchema()
    display(df.limit(5))


for t in [tables.ratings, tables.movies, tables.links, tables.scraped]:
    summarize(t)
