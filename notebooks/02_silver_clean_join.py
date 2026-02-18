# Databricks notebook source
from __future__ import annotations

from pyspark.sql import functions as F

BRONZE = "workspace.movielens_bronze"
SILVER = "workspace.movielens_silver"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER}")

# --- Read Bronze ---
ratings_b = spark.table(f"{BRONZE}.bronze_ratings")
movies_b  = spark.table(f"{BRONZE}.bronze_movies")
links_b   = spark.table(f"{BRONZE}.bronze_links")
tags_b    = spark.table(f"{BRONZE}.bronze_tags")

# -----------------------------
# Silver Movies
# -----------------------------
movies_s = (
    movies_b
    .select(
        F.col("movieId").cast("long").alias("movieId"),
        F.col("title").cast("string").alias("title"),
        F.col("genres").cast("string").alias("genres_raw"),
    )
    .withColumn("genres_raw", F.when(F.col("genres_raw") == "(no genres listed)", F.lit("")).otherwise(F.col("genres_raw")))
    .withColumn(
        "genres",
        F.when(
            (F.col("genres_raw").isNull()) | (F.col("genres_raw") == ""),
            F.array().cast("array<string>")
        ).otherwise(F.split("genres_raw", "\\|"))
    )
    .drop("genres_raw")
    .dropDuplicates(["movieId"])  # enforce 1 row per movieId
)

# -----------------------------
# Silver Ratings
# -----------------------------
ratings_s = (
    ratings_b
    .select(
        F.col("userId").cast("long").alias("userId"),
        F.col("movieId").cast("long").alias("movieId"),
        F.col("rating").cast("double").alias("rating"),
        F.col("timestamp").cast("long").alias("ts"),
    )
    .filter(F.col("rating").isNotNull())
    .filter((F.col("rating") >= F.lit(0.5)) & (F.col("rating") <= F.lit(5.0)))
)

# -----------------------------
# Silver Links (dedupe per movieId)
# -----------------------------
links_s = (
    links_b
    .select(
        F.col("movieId").cast("long").alias("movieId"),
        F.col("imdbId").cast("string").alias("imdbId"),
        F.col("tmdbId").cast("string").alias("tmdbId"),
    )
    .dropDuplicates(["movieId"])
)

# -----------------------------
# Silver Tags
# -----------------------------
tags_s = (
    tags_b
    .select(
        F.col("userId").cast("long").alias("userId"),
        F.col("movieId").cast("long").alias("movieId"),
        F.col("tag").cast("string").alias("tag"),
        F.col("timestamp").cast("long").alias("ts"),
    )
    .filter(F.col("tag").isNotNull())
    .withColumn("tag", F.trim(F.col("tag")))
    .filter(F.length(F.col("tag")) > 0)
)

# -----------------------------
# Coverage check (ratings -> movies)
# -----------------------------
ratings_without_movie = (
    ratings_s.alias("r")
    .join(movies_s.alias("m"), on="movieId", how="left")
    .filter(F.col("m.title").isNull())
    .select("r.*")
)

missing_count = ratings_without_movie.count()
total_count = ratings_s.count()

pct_missing = (missing_count / total_count) if total_count else 0.0
print(f"Ratings total: {total_count}")
print(f"Ratings without movie: {missing_count} ({pct_missing:.6%})")

# Quarantine table (audit)
ratings_without_movie.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{SILVER}.quarantine_ratings_missing_movie")

# Keep only valid ratings for downstream
ratings_s_valid = (
    ratings_s.alias("r")
    .join(movies_s.select("movieId").alias("m"), on="movieId", how="inner")
    .select("r.*")
)

# -----------------------------
# Quality gates (fail fast if broken)
# -----------------------------
movies_unique = movies_s.select("movieId").distinct().count()
movies_total = movies_s.count()
if movies_unique != movies_total:
    raise RuntimeError(f"Quality gate failed: movies movieId not unique ({movies_unique} distinct vs {movies_total} rows)")

links_unique = links_s.select("movieId").distinct().count()
links_total = links_s.count()
if links_unique != links_total:
    raise RuntimeError(f"Quality gate failed: links movieId not unique ({links_unique} distinct vs {links_total} rows)")

if pct_missing > 0.001:  # >0.1%
    raise RuntimeError(f"Quality gate failed: ratings->movies missing join too high: {pct_missing:.6%}")

# -----------------------------
# Write Silver (SCD1 overwrite)
# -----------------------------
def write(df, name: str) -> None:
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SILVER}.{name}")

write(movies_s, "silver_movies")
write(ratings_s_valid, "silver_ratings")
write(links_s, "silver_links")
write(tags_s, "silver_tags")

# -----------------------------
# Summary
# -----------------------------
for t in ["silver_movies", "silver_ratings", "silver_links", "silver_tags", "quarantine_ratings_missing_movie"]:
    c = spark.table(f"{SILVER}.{t}").count()
    print(f"{SILVER}.{t}: {c}")

