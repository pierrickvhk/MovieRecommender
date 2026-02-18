# Design: 4 Databricks Workflow Notebooks

**Date:** 2026-02-18
**Branch:** scraping
**Deadline:** 2026-02-20

## Summary

Port the existing local PySpark notebooks into 4 Databricks-ready notebooks that chain together as a single Databricks Workflow: Ingest → Clean → Transform → Train.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Storage format | Delta Lake | Native to Databricks; required by instructions spec |
| Table storage | Managed Delta tables in Unity Catalog | Best for SQL queries, evaluation demos, most Databricks-native |
| File storage | Unity Catalog Volume | Modern Databricks approach for raw files |
| TMDB credentials | `dbutils.secrets.get()` | Secure, no credentials in code |
| SparkSession | Pre-existing `spark` | Standard on Databricks clusters |

## Global Conventions

- No `SparkSession.builder` — `spark` is already available on Databricks
- Each notebook starts with `dbutils.widgets` for `catalog`, `schema`, and where needed `volume_path`
- All table writes: `spark.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.{table}")`
- Fact tables partitioned by a date column
- Every notebook is idempotent (safe to re-run)
- Each notebook ends with a summary print of row counts

## Notebook 1: `01_ingest.ipynb` (Bronze Layer)

**Purpose:** Read raw CSVs from a Unity Catalog Volume and scrape TMDB API. Write 5 Bronze Delta tables.

**Widgets:** `catalog`, `schema`, `volume_path`, `secret_scope`

**Steps:**
1. Read 4 CSVs from `{volume_path}/ml-32m/` with explicit schemas (ratings, movies, links, tags)
2. Add `_ingestion_timestamp` and `_source_file` metadata columns
3. Write `bronze_ratings`, `bronze_movies`, `bronze_links`, `bronze_tags` as managed Delta tables
4. Identify top-500 movies by rating count from bronze data
5. Scrape TMDB API for those 500 movies using `dbutils.secrets.get(secret_scope, "TMDB_BEARER_TOKEN")`
6. Write `bronze_enrichment` as managed Delta table
7. Print row counts for all 5 tables

**Source schemas:** Identical to `ingestion.ipynb` — explicit StructType for all 4 CSVs; enrichment uses the schema from `scraping.ipynb`.

## Notebook 2: `02_clean.ipynb` (Silver Layer)

**Purpose:** Clean, deduplicate, enforce types, and join Bronze tables. Write 5 Silver Delta tables.

**Widgets:** `catalog`, `schema`

**Logic (identical to `silver.ipynb`):**
- Ratings: drop metadata cols, Unix epoch → TimestampType (`rated_at`), dedup on `(userId, movieId, rated_at)`, null filter, range filter `[0.5–5.0]`
- Movies: dedup on `movieId`, genres pipe-split → `ArrayType`, year regex extract, `clean_title` without trailing year
- Links: dedup, `imdbId` zero-padded to 7 chars
- Tags: Unix epoch → `tagged_at`, lowercase+trim tag, dedup, null filter
- Referential integrity: inner-join ratings and tags against valid movieIds
- `silver_movies_with_links`: LEFT JOIN movies + links

**Writes:** `silver_ratings`, `silver_movies`, `silver_links`, `silver_tags`, `silver_movies_with_links`

## Notebook 3: `03_transform.ipynb` (Gold Layer)

**Purpose:** Build star-schema tables for analytics and ML training.

**Widgets:** `catalog`, `schema`

**Tables built (logic identical to `gold.ipynb`):**

- `gold_fact_ratings`: userId, movieId, rating, rated_at, rating_year — partitioned by `rating_year`; assert 0 nulls on key columns
- `gold_dim_users`: rating aggregates (count, avg, stddev, min, max), genre breadth (via explode), tag count, active years, `is_power_user` flag (≥500 ratings, ≥5 genres, ≥1 tag)
- `gold_dim_movies_enriched`: movies + links + rating aggs + tag counts + TMDB enrichment; `budget`/`revenue` 0→null; `profit` = revenue − budget; `has_enrichment` flag

**Validation:** Null audit across all 3 Gold tables + print "Highest Rated Director" query.

## Notebook 4: `04_train.ipynb` (ALS Model)

**Purpose:** Train an ALS recommendation model, evaluate it, save it, and generate top-10 recommendations per user.

**Widgets:** `catalog`, `schema`, `volume_path`, `als_rank` (default 10), `als_max_iter` (default 10), `als_reg_param` (default 0.1)

**Steps:**
1. Read `gold_fact_ratings`; extract `(userId, movieId, rating)` as FloatType
2. 80/20 train/test split (seed=42); cache training set
3. Train `ALS(coldStartStrategy="drop", implicitPrefs=False)` with widget hyperparameters
4. Evaluate on test set — compute and print RMSE
5. Save trained model to `{volume_path}/models/als_model/`
6. Generate top-10 recommendations for all users via `model.recommendForAllUsers(10)`
7. Explode to `(userId, movieId, predicted_rating)`; join `gold_dim_movies_enriched` for movie titles
8. Write `gold_recommendations` as managed Delta table
9. Demo: show top-10 recommendations for a sample userId

## Databricks Workflow Chain

```
01_ingest → 02_clean → 03_transform → 04_train
```

Each notebook is a sequential task in the workflow. Catalog/schema/volume_path are configured once as job-level parameters and passed to each notebook as widget values.
