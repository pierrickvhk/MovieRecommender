# Databricks notebook source
from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ============================================================
# 03_gold_features.py
# Gold = curated tables for analytics + model training (ALS)
# Serverless-safe, idempotent overwrites
# ============================================================

SILVER = "workspace.movielens_silver"
GOLD = "workspace.movielens_gold"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD}")

# --- Read Silver ---
ratings = spark.table(
    f"{SILVER}.silver_ratings"
)  # userId(long), movieId(long), rating(double), ts(long)
movies = spark.table(
    f"{SILVER}.silver_movies"
)  # movieId(long), title(string), genres(array<string>)
links = spark.table(
    f"{SILVER}.silver_links"
)  # movieId(long), imdbId(string), tmdbId(string)
tags = spark.table(
    f"{SILVER}.silver_tags"
)  # userId(long), movieId(long), tag(string), ts(long)


# Helper writer
def write(df, name: str, partition_cols: list[str] | None = None) -> None:
    w = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_cols:
        w = w.partitionBy(*partition_cols)
    w.saveAsTable(f"{GOLD}.{name}")


# -----------------------------
# 1) fact_ratings
# - Add deterministic partition bucket to improve reads
# -----------------------------
fact_ratings = ratings.select(
    F.col("userId").cast("long").alias("userId"),
    F.col("movieId").cast("long").alias("movieId"),
    F.col("rating").cast("double").alias("rating"),
    F.col("ts").cast("long").alias("ts"),
).withColumn("user_bucket", (F.col("userId") % F.lit(50)).cast("int"))

write(fact_ratings, "fact_ratings", partition_cols=["user_bucket"])

# -----------------------------
# 2) dim_users
# - Basic user behavior profile
# -----------------------------
dim_users = (
    fact_ratings.groupBy("userId")
    .agg(
        F.count("*").alias("n_ratings"),
        F.avg("rating").alias("avg_rating"),
        F.stddev_pop("rating").alias("rating_std"),
        F.max("ts").alias("last_ts"),
        F.min("ts").alias("first_ts"),
    )
    .withColumn("is_power_user", F.col("n_ratings") >= F.lit(200))
)

write(dim_users, "dim_users")

# -----------------------------
# 3) dim_movies_enriched
# - movies + links (later can be enriched via scraped JSON)
# -----------------------------
dim_movies_enriched = (
    movies.alias("m")
    .join(links.alias("l"), on="movieId", how="left")
    .select(
        F.col("m.movieId").cast("long").alias("movieId"),
        F.col("m.title").cast("string").alias("title"),
        F.col("m.genres").alias("genres"),
        F.col("l.imdbId").cast("string").alias("imdbId"),
        F.col("l.tmdbId").cast("string").alias("tmdbId"),
        # placeholders for scraped fields
        F.lit(None).cast("string").alias("director"),
        F.lit(None).cast("string").alias("budget"),
        F.lit(None).cast("string").alias("poster_url"),
    )
)

write(dim_movies_enriched, "dim_movies_enriched")

# -----------------------------
# 4) agg_movie_tags
# - tags aggregated per movie
# -----------------------------
agg_movie_tags = tags.groupBy("movieId").agg(
    F.count("*").alias("n_tags"),
    F.collect_set("tag").alias("tags_set"),
)

write(agg_movie_tags, "agg_movie_tags")

# -----------------------------
# 5) train_view_als (with ts)
# - ALS expects user,item,rating (and we keep ts for time split)
# - Cast to int/float for MLlib ALS
# -----------------------------
train_view_als = fact_ratings.select(
    F.col("userId").cast("int").alias("userId"),
    F.col("movieId").cast("int").alias("movieId"),
    F.col("rating").cast("float").alias("rating"),
    F.col("ts").cast("long").alias("ts"),
)

write(train_view_als, "train_view_als")

# -----------------------------
# 6) ALS splits (time-based per user)
# - last interaction per user -> test
# - 2nd last -> val
# - rest -> train
# -----------------------------
w = Window.partitionBy("userId").orderBy(F.col("ts").desc(), F.col("movieId").desc())
ranked = train_view_als.withColumn("rn", F.row_number().over(w))

als_test = ranked.filter(F.col("rn") == 1).select("userId", "movieId", "rating")
als_val = ranked.filter(F.col("rn") == 2).select("userId", "movieId", "rating")
als_train = ranked.filter(F.col("rn") >= 3).select("userId", "movieId", "rating")

write(als_train, "als_train")
write(als_val, "als_val")
write(als_test, "als_test")

# -----------------------------
# 7) Summary
# -----------------------------
print("=== Gold summary counts ===")
for t in [
    "fact_ratings",
    "dim_users",
    "dim_movies_enriched",
    "agg_movie_tags",
    "train_view_als",
    "als_train",
    "als_val",
    "als_test",
]:
    c = spark.table(f"{GOLD}.{t}").count()
    print(f"{GOLD}.{t}: {c}")

print("\n=== ALS split ratio (sanity) ===")
train_c = spark.table(f"{GOLD}.als_train").count()
val_c = spark.table(f"{GOLD}.als_val").count()
test_c = spark.table(f"{GOLD}.als_test").count()
total = train_c + val_c + test_c
print(f"total(split): {total}, train: {train_c}, val: {val_c}, test: {test_c}")

print("\nSample dim_users:")
display(spark.table(f"{GOLD}.dim_users").orderBy(F.col("n_ratings").desc()).limit(5))

print("\nTop movies by n_tags (joined to titles):")
top_tags = (
    spark.table(f"{GOLD}.agg_movie_tags")
    .orderBy(F.col("n_tags").desc())
    .limit(10)
    .join(
        spark.table(f"{GOLD}.dim_movies_enriched").select("movieId", "title"),
        on="movieId",
        how="left",
    )
)
display(top_tags)