# Databricks Workflow Notebooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create 4 self-contained Databricks notebooks (Ingest → Clean → Transform → Train) that chain together as a Databricks Workflow, porting the existing local PySpark notebooks to use Unity Catalog Delta tables.

**Architecture:** Each notebook reads from a previous layer's managed Delta tables and writes the next layer using `spark.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.{table}")`. Parameters (catalog, schema, volume_path) are passed via `dbutils.widgets`. No `SparkSession.builder` — Databricks provides `spark` automatically.

**Tech Stack:** PySpark 3.x (Databricks runtime), Delta Lake, Unity Catalog, Databricks Secrets, Spark MLlib ALS

---

## Context: Key Databricks Differences vs Local Notebooks

| Local | Databricks |
|---|---|
| `SparkSession.builder.master("local[*]").getOrCreate()` | `spark` already exists — do NOT create one |
| `bronze/ratings` (relative path) | `{catalog}.{schema}.bronze_ratings` (managed Delta table) |
| `load_dotenv(); os.getenv("TMDB_BEARER_TOKEN")` | `dbutils.secrets.get(scope, "TMDB_BEARER_TOKEN")` |
| `.write.mode("overwrite").parquet(path)` | `.write.format("delta").mode("overwrite").saveAsTable(full_name)` |
| `PROJECT = "/local/path"` | `volume_path` widget (e.g. `/Volumes/main/movielens/data`) |

## Output Files

All 4 notebooks go in `databricks/` at the repo root:
- `databricks/01_ingest.ipynb`
- `databricks/02_clean.ipynb`
- `databricks/03_transform.ipynb`
- `databricks/04_train.ipynb`

---

## Task 1: Create `databricks/01_ingest.ipynb` (Bronze Layer)

**Files:**
- Create: `databricks/01_ingest.ipynb`

This notebook combines `ingestion.ipynb` + `scraping.ipynb`. It reads raw CSVs from a Unity Catalog Volume and scrapes TMDB, writing 5 Bronze Delta tables.

### Step 1: Create the notebook file

Create `databricks/01_ingest.ipynb` as a new Jupyter notebook. Use the `Write` tool with the full JSON content below.

**Full notebook content:**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 01 — Ingest (Bronze Layer)\n", "\n", "Reads raw CSVs from a Unity Catalog Volume and scrapes TMDB API.\n", "Writes 5 Bronze managed Delta tables: `bronze_ratings`, `bronze_movies`, `bronze_links`, `bronze_tags`, `bronze_enrichment`."]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Parameters — set as Databricks Workflow job parameters\n",
    "dbutils.widgets.text(\"catalog\",      \"main\",                   \"Unity Catalog name\")\n",
    "dbutils.widgets.text(\"schema\",       \"movielens\",              \"Schema / database name\")\n",
    "dbutils.widgets.text(\"volume_path\",  \"/Volumes/main/movielens/data\", \"Volume root path\")\n",
    "dbutils.widgets.text(\"secret_scope\", \"movielens\",              \"Databricks secret scope\")\n",
    "\n",
    "CATALOG     = dbutils.widgets.get(\"catalog\")\n",
    "SCHEMA      = dbutils.widgets.get(\"schema\")\n",
    "VOLUME      = dbutils.widgets.get(\"volume_path\")\n",
    "SECRET_SCOPE = dbutils.widgets.get(\"secret_scope\")\n",
    "\n",
    "RAW_PATH = f\"{VOLUME}/ml-32m\"\n",
    "\n",
    "def tbl(name):\n",
    "    return f\"{CATALOG}.{SCHEMA}.{name}\"\n",
    "\n",
    "print(f\"Catalog : {CATALOG}\")\n",
    "print(f\"Schema  : {SCHEMA}\")\n",
    "print(f\"Volume  : {VOLUME}\")\n",
    "print(f\"Raw CSVs: {RAW_PATH}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Ensure target schema exists\n",
    "spark.sql(f\"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}\")\n",
    "print(f\"Schema {CATALOG}.{SCHEMA} ready.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType, ArrayType\n",
    "from pyspark.sql.functions import current_timestamp, lit\n",
    "\n",
    "ratings_schema = StructType([\n",
    "    StructField(\"userId\",    IntegerType(), True),\n",
    "    StructField(\"movieId\",   IntegerType(), True),\n",
    "    StructField(\"rating\",    FloatType(),   True),\n",
    "    StructField(\"timestamp\", LongType(),    True),\n",
    "])\n",
    "movies_schema = StructType([\n",
    "    StructField(\"movieId\", IntegerType(), True),\n",
    "    StructField(\"title\",   StringType(),  True),\n",
    "    StructField(\"genres\",  StringType(),  True),\n",
    "])\n",
    "links_schema = StructType([\n",
    "    StructField(\"movieId\", IntegerType(), True),\n",
    "    StructField(\"imdbId\",  IntegerType(), True),\n",
    "    StructField(\"tmdbId\",  IntegerType(), True),\n",
    "])\n",
    "tags_schema = StructType([\n",
    "    StructField(\"userId\",    IntegerType(), True),\n",
    "    StructField(\"movieId\",   IntegerType(), True),\n",
    "    StructField(\"tag\",       StringType(),  True),\n",
    "    StructField(\"timestamp\", LongType(),    True),\n",
    "])\n",
    "print(\"Schemas defined.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "def add_metadata(df, source_name):\n",
    "    return df \\\n",
    "        .withColumn(\"_ingestion_timestamp\", current_timestamp()) \\\n",
    "        .withColumn(\"_source_file\", lit(source_name))\n",
    "\n",
    "df_ratings = spark.read.csv(f\"{RAW_PATH}/ratings.csv\", header=True, schema=ratings_schema)\n",
    "df_movies  = spark.read.csv(f\"{RAW_PATH}/movies.csv\",  header=True, schema=movies_schema)\n",
    "df_links   = spark.read.csv(f\"{RAW_PATH}/links.csv\",   header=True, schema=links_schema)\n",
    "df_tags    = spark.read.csv(f\"{RAW_PATH}/tags.csv\",    header=True, schema=tags_schema)\n",
    "\n",
    "add_metadata(df_ratings, \"ratings.csv\").write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"bronze_ratings\"))\n",
    "add_metadata(df_movies,  \"movies.csv\" ).write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"bronze_movies\"))\n",
    "add_metadata(df_links,   \"links.csv\"  ).write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"bronze_links\"))\n",
    "add_metadata(df_tags,    \"tags.csv\"   ).write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"bronze_tags\"))\n",
    "print(\"Bronze CSV tables written.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Identify top-500 movies by rating count for TMDB scraping\n",
    "from pyspark.sql.functions import col, count, avg, round as spark_round\n",
    "\n",
    "bronze_ratings = spark.table(tbl(\"bronze_ratings\"))\n",
    "bronze_links   = spark.table(tbl(\"bronze_links\"))\n",
    "bronze_movies  = spark.table(tbl(\"bronze_movies\"))\n",
    "\n",
    "top_500 = (\n",
    "    bronze_ratings\n",
    "    .groupBy(\"movieId\")\n",
    "    .agg(count(\"*\").alias(\"rating_count\"), spark_round(avg(\"rating\"), 2).alias(\"avg_rating\"))\n",
    "    .orderBy(col(\"rating_count\").desc())\n",
    "    .limit(500)\n",
    ")\n",
    "\n",
    "movies_to_scrape = (\n",
    "    top_500\n",
    "    .join(bronze_links,  \"movieId\")\n",
    "    .join(bronze_movies, \"movieId\")\n",
    "    .select(\"movieId\", \"title\", \"tmdbId\", \"imdbId\", \"rating_count\", \"avg_rating\")\n",
    "    .filter(col(\"tmdbId\").isNotNull())\n",
    "    .orderBy(col(\"rating_count\").desc())\n",
    "    .collect()\n",
    ")\n",
    "print(f\"Movies to scrape: {len(movies_to_scrape)}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import os, time, requests\n",
    "\n",
    "TMDB_TOKEN = dbutils.secrets.get(scope=SECRET_SCOPE, key=\"TMDB_BEARER_TOKEN\")\n",
    "\n",
    "session = requests.Session()\n",
    "session.headers.update({\n",
    "    \"Authorization\": f\"Bearer {TMDB_TOKEN}\",\n",
    "    \"accept\": \"application/json\",\n",
    "})\n",
    "\n",
    "def fetch_movie(tmdb_id):\n",
    "    url = f\"https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=credits\"\n",
    "    while True:\n",
    "        resp = session.get(url, timeout=10)\n",
    "        if resp.status_code == 429:\n",
    "            time.sleep(int(resp.headers.get(\"Retry-After\", 2)))\n",
    "            continue\n",
    "        if resp.status_code != 200:\n",
    "            print(f\"  WARN: tmdb_id={tmdb_id} -> {resp.status_code}\")\n",
    "            return None\n",
    "        return resp.json()\n",
    "\n",
    "def extract(raw, movie_id):\n",
    "    poster = raw.get(\"poster_path\")\n",
    "    return {\n",
    "        \"movieId\":           movie_id,\n",
    "        \"tmdbId\":            raw.get(\"id\"),\n",
    "        \"title\":             raw.get(\"title\"),\n",
    "        \"directors\":         [m[\"name\"] for m in raw.get(\"credits\", {}).get(\"crew\", []) if m.get(\"job\") == \"Director\"],\n",
    "        \"budget\":            raw.get(\"budget\"),\n",
    "        \"revenue\":           raw.get(\"revenue\"),\n",
    "        \"runtime\":           raw.get(\"runtime\"),\n",
    "        \"release_date\":      raw.get(\"release_date\"),\n",
    "        \"poster_url\":        f\"https://image.tmdb.org/t/p/w500{poster}\" if poster else None,\n",
    "        \"overview\":          raw.get(\"overview\"),\n",
    "        \"vote_average\":      raw.get(\"vote_average\"),\n",
    "        \"original_language\": raw.get(\"original_language\"),\n",
    "    }\n",
    "\n",
    "results = []\n",
    "total = len(movies_to_scrape)\n",
    "for i, row in enumerate(movies_to_scrape):\n",
    "    raw = fetch_movie(row.tmdbId)\n",
    "    if raw:\n",
    "        results.append(extract(raw, row.movieId))\n",
    "    time.sleep(0.05)\n",
    "    if (i + 1) % 50 == 0:\n",
    "        print(f\"  [{i+1}/{total}] scraped so far: {len(results)}\")\n",
    "\n",
    "print(f\"Scraped {len(results)}/{total} movies.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "enrichment_schema = StructType([\n",
    "    StructField(\"movieId\",           IntegerType(), True),\n",
    "    StructField(\"tmdbId\",            IntegerType(), True),\n",
    "    StructField(\"title\",             StringType(),  True),\n",
    "    StructField(\"directors\",         ArrayType(StringType()), True),\n",
    "    StructField(\"budget\",            LongType(),    True),\n",
    "    StructField(\"revenue\",           LongType(),    True),\n",
    "    StructField(\"runtime\",           IntegerType(), True),\n",
    "    StructField(\"release_date\",      StringType(),  True),\n",
    "    StructField(\"poster_url\",        StringType(),  True),\n",
    "    StructField(\"overview\",          StringType(),  True),\n",
    "    StructField(\"vote_average\",      FloatType(),   True),\n",
    "    StructField(\"original_language\", StringType(),  True),\n",
    "])\n",
    "\n",
    "(\n",
    "    spark.createDataFrame(results, enrichment_schema)\n",
    "    .withColumn(\"_ingestion_timestamp\", current_timestamp())\n",
    "    .withColumn(\"_source_file\", lit(\"tmdb_api\"))\n",
    "    .write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"bronze_enrichment\"))\n",
    ")\n",
    "print(\"bronze_enrichment written.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Verification\n",
    "tables = [\"bronze_ratings\", \"bronze_movies\", \"bronze_links\", \"bronze_tags\", \"bronze_enrichment\"]\n",
    "print(\"=== Bronze Row Counts ===\")\n",
    "for t in tables:\n",
    "    n = spark.table(tbl(t)).count()\n",
    "    print(f\"  {t:<25} {n:>12,}\")"
   ],
   "outputs": [],
   "execution_count": null
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

### Step 2: Verify expected output shape

After running in Databricks, the final cell should print:
```
=== Bronze Row Counts ===
  bronze_ratings              32,000,204
  bronze_movies                   87,585
  bronze_links                    87,585
  bronze_tags                  2,000,072
  bronze_enrichment                  500
```

### Step 3: Commit

```bash
git add databricks/01_ingest.ipynb
git commit -m "feat: add 01_ingest Databricks notebook (Bronze layer)"
```

---

## Task 2: Create `databricks/02_clean.ipynb` (Silver Layer)

**Files:**
- Create: `databricks/02_clean.ipynb`

Ports `silver.ipynb` exactly — reads Bronze Delta tables, applies cleaning logic, writes 5 Silver Delta tables.

### Step 1: Create the notebook file

**Full notebook content:**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 02 — Clean (Silver Layer)\n", "\n", "Reads Bronze Delta tables, applies type casting, deduplication, null filtering, and referential integrity.\n", "Writes 5 Silver managed Delta tables."]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "dbutils.widgets.text(\"catalog\", \"main\",      \"Unity Catalog name\")\n",
    "dbutils.widgets.text(\"schema\",  \"movielens\", \"Schema / database name\")\n",
    "\n",
    "CATALOG = dbutils.widgets.get(\"catalog\")\n",
    "SCHEMA  = dbutils.widgets.get(\"schema\")\n",
    "\n",
    "def tbl(name):\n",
    "    return f\"{CATALOG}.{SCHEMA}.{name}\"\n",
    "\n",
    "print(f\"Reading from / writing to: {CATALOG}.{SCHEMA}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "from pyspark.sql.functions import (\n",
    "    col, from_unixtime, split, when, array,\n",
    "    regexp_extract, regexp_replace, trim, lpad, lower,\n",
    ")\n",
    "\n",
    "bronze_ratings = spark.table(tbl(\"bronze_ratings\"))\n",
    "bronze_movies  = spark.table(tbl(\"bronze_movies\"))\n",
    "bronze_links   = spark.table(tbl(\"bronze_links\"))\n",
    "bronze_tags    = spark.table(tbl(\"bronze_tags\"))\n",
    "print(\"Bronze tables loaded.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "silver_ratings = (\n",
    "    bronze_ratings\n",
    "    .drop(\"_ingestion_timestamp\", \"_source_file\")\n",
    "    .withColumn(\"rated_at\", from_unixtime(col(\"timestamp\")).cast(\"timestamp\"))\n",
    "    .drop(\"timestamp\")\n",
    "    .dropDuplicates([\"userId\", \"movieId\", \"rated_at\"])\n",
    "    .filter(col(\"userId\").isNotNull() & col(\"movieId\").isNotNull() & col(\"rating\").isNotNull())\n",
    "    .filter((col(\"rating\") >= 0.5) & (col(\"rating\") <= 5.0))\n",
    ")\n",
    "print(f\"silver_ratings: {silver_ratings.count():,} rows\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "silver_movies = (\n",
    "    bronze_movies\n",
    "    .drop(\"_ingestion_timestamp\", \"_source_file\")\n",
    "    .dropDuplicates([\"movieId\"])\n",
    "    .withColumn(\n",
    "        \"genres\",\n",
    "        when(col(\"genres\") == \"(no genres listed)\", array())\n",
    "        .otherwise(split(col(\"genres\"), \"\\\\|\"))\n",
    "    )\n",
    "    .withColumn(\"year\", regexp_extract(col(\"title\"), r\"\\((\\d{4})\\)\\s*$\", 1).cast(\"int\"))\n",
    "    .withColumn(\"clean_title\", trim(regexp_replace(col(\"title\"), r\"\\s*\\(\\d{4}\\)\\s*$\", \"\")))\n",
    ")\n",
    "print(f\"silver_movies: {silver_movies.count():,} rows\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "silver_links = (\n",
    "    bronze_links\n",
    "    .drop(\"_ingestion_timestamp\", \"_source_file\")\n",
    "    .dropDuplicates([\"movieId\"])\n",
    "    .filter(col(\"movieId\").isNotNull())\n",
    "    .withColumn(\"imdbId\", lpad(col(\"imdbId\").cast(\"string\"), 7, \"0\"))\n",
    ")\n",
    "\n",
    "silver_tags = (\n",
    "    bronze_tags\n",
    "    .drop(\"_ingestion_timestamp\", \"_source_file\")\n",
    "    .withColumn(\"tagged_at\", from_unixtime(col(\"timestamp\")).cast(\"timestamp\"))\n",
    "    .drop(\"timestamp\")\n",
    "    .withColumn(\"tag\", trim(lower(col(\"tag\"))))\n",
    "    .filter(col(\"tag\").isNotNull() & (col(\"tag\") != \"\"))\n",
    "    .dropDuplicates([\"userId\", \"movieId\", \"tag\", \"tagged_at\"])\n",
    ")\n",
    "print(f\"silver_links: {silver_links.count():,} rows\")\n",
    "print(f\"silver_tags:  {silver_tags.count():,} rows\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Referential integrity — drop ratings/tags for movies not in silver_movies\n",
    "valid_movie_ids = silver_movies.select(\"movieId\")\n",
    "silver_ratings = silver_ratings.join(valid_movie_ids, \"movieId\", \"inner\")\n",
    "silver_tags    = silver_tags.join(valid_movie_ids,    \"movieId\", \"inner\")\n",
    "\n",
    "# Master join\n",
    "silver_movies_with_links = silver_movies.join(silver_links, \"movieId\", \"left\")\n",
    "print(\"Referential integrity enforced. Master join complete.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "silver_ratings          .write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"silver_ratings\"))\n",
    "silver_movies           .write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"silver_movies\"))\n",
    "silver_links            .write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"silver_links\"))\n",
    "silver_tags             .write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"silver_tags\"))\n",
    "silver_movies_with_links.write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"silver_movies_with_links\"))\n",
    "print(\"All Silver tables written.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Verification\n",
    "silver_tables = {\n",
    "    \"silver_ratings\":           spark.table(tbl(\"silver_ratings\")),\n",
    "    \"silver_movies\":            spark.table(tbl(\"silver_movies\")),\n",
    "    \"silver_links\":             spark.table(tbl(\"silver_links\")),\n",
    "    \"silver_tags\":              spark.table(tbl(\"silver_tags\")),\n",
    "    \"silver_movies_with_links\": spark.table(tbl(\"silver_movies_with_links\")),\n",
    "}\n",
    "print(\"=== Silver Row Counts ===\")\n",
    "for name, df in silver_tables.items():\n",
    "    print(f\"  {name:<30} {df.count():>12,}\")"
   ],
   "outputs": [],
   "execution_count": null
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

### Step 2: Verify expected output shape

```
=== Silver Row Counts ===
  silver_ratings                  32,000,204
  silver_movies                       87,585
  silver_links                        87,585
  silver_tags                      2,000,072
  silver_movies_with_links            87,585
```

### Step 3: Commit

```bash
git add databricks/02_clean.ipynb
git commit -m "feat: add 02_clean Databricks notebook (Silver layer)"
```

---

## Task 3: Create `databricks/03_transform.ipynb` (Gold Layer)

**Files:**
- Create: `databricks/03_transform.ipynb`

Ports `gold.ipynb` — builds `gold_fact_ratings` (partitioned), `gold_dim_users`, and `gold_dim_movies_enriched`, with null audit and Highest Rated Director query.

### Step 1: Create the notebook file

**Full notebook content:**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 03 — Transform (Gold Layer)\n", "\n", "Builds star-schema tables for analytics and ML training.\n", "- `gold_fact_ratings` — 32M rows, partitioned by rating_year\n", "- `gold_dim_users` — ~200K rows with behavioral aggregates and Power User flag\n", "- `gold_dim_movies_enriched` — 87.5K rows with TMDB metadata"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "dbutils.widgets.text(\"catalog\", \"main\",      \"Unity Catalog name\")\n",
    "dbutils.widgets.text(\"schema\",  \"movielens\", \"Schema / database name\")\n",
    "\n",
    "CATALOG = dbutils.widgets.get(\"catalog\")\n",
    "SCHEMA  = dbutils.widgets.get(\"schema\")\n",
    "\n",
    "def tbl(name):\n",
    "    return f\"{CATALOG}.{SCHEMA}.{name}\"\n",
    "\n",
    "print(f\"Reading from / writing to: {CATALOG}.{SCHEMA}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "from pyspark.sql.functions import (\n",
    "    col, year as spark_year, count, avg, stddev,\n",
    "    min as spark_min, max as spark_max,\n",
    "    countDistinct, explode, when, lit,\n",
    "    round as spark_round, sum as spark_sum,\n",
    ")\n",
    "\n",
    "silver_ratings           = spark.table(tbl(\"silver_ratings\"))\n",
    "silver_movies            = spark.table(tbl(\"silver_movies\"))\n",
    "silver_tags              = spark.table(tbl(\"silver_tags\"))\n",
    "silver_movies_with_links = spark.table(tbl(\"silver_movies_with_links\"))\n",
    "bronze_enrichment        = spark.table(tbl(\"bronze_enrichment\"))\n",
    "print(\"Source tables loaded.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# ── fact_ratings ──────────────────────────────────────────────────────────────\n",
    "fact_ratings = (\n",
    "    silver_ratings\n",
    "    .select(\"userId\", \"movieId\", \"rating\", \"rated_at\")\n",
    "    .withColumn(\"rating_year\", spark_year(\"rated_at\"))\n",
    ")\n",
    "\n",
    "for c in [\"userId\", \"movieId\", \"rating\"]:\n",
    "    null_count = fact_ratings.filter(col(c).isNull()).count()\n",
    "    assert null_count == 0, f\"Unexpected nulls in fact_ratings.{c}: {null_count}\"\n",
    "    print(f\"  {c}: 0 nulls ✓\")\n",
    "\n",
    "(\n",
    "    fact_ratings.write\n",
    "    .format(\"delta\")\n",
    "    .mode(\"overwrite\")\n",
    "    .partitionBy(\"rating_year\")\n",
    "    .saveAsTable(tbl(\"gold_fact_ratings\"))\n",
    ")\n",
    "print(f\"gold_fact_ratings: {fact_ratings.count():,} rows written.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# ── dim_users ─────────────────────────────────────────────────────────────────\n",
    "user_rating_aggs = (\n",
    "    silver_ratings\n",
    "    .groupBy(\"userId\")\n",
    "    .agg(\n",
    "        count(\"*\").alias(\"rating_count\"),\n",
    "        spark_round(avg(\"rating\"), 2).alias(\"avg_rating\"),\n",
    "        spark_round(stddev(\"rating\"), 2).alias(\"rating_stddev\"),\n",
    "        spark_min(\"rating\").alias(\"min_rating\"),\n",
    "        spark_max(\"rating\").alias(\"max_rating\"),\n",
    "        countDistinct(\"movieId\").alias(\"distinct_movies_rated\"),\n",
    "        spark_min(\"rated_at\").alias(\"first_rating_at\"),\n",
    "        spark_max(\"rated_at\").alias(\"last_rating_at\"),\n",
    "    )\n",
    ")\n",
    "\n",
    "user_genre_breadth = (\n",
    "    silver_ratings.select(\"userId\", \"movieId\")\n",
    "    .join(silver_movies.select(\"movieId\", \"genres\"), \"movieId\")\n",
    "    .select(\"userId\", explode(\"genres\").alias(\"genre\"))\n",
    "    .groupBy(\"userId\")\n",
    "    .agg(countDistinct(\"genre\").alias(\"distinct_genres_rated\"))\n",
    ")\n",
    "\n",
    "user_tag_counts = (\n",
    "    silver_tags\n",
    "    .groupBy(\"userId\")\n",
    "    .agg(count(\"*\").alias(\"tag_count\"))\n",
    ")\n",
    "\n",
    "dim_users = (\n",
    "    user_rating_aggs\n",
    "    .join(user_genre_breadth, \"userId\", \"left\")\n",
    "    .join(user_tag_counts,    \"userId\", \"left\")\n",
    "    .fillna(0, subset=[\"distinct_genres_rated\", \"tag_count\"])\n",
    "    .withColumn(\"active_years\", spark_year(\"last_rating_at\") - spark_year(\"first_rating_at\") + 1)\n",
    "    .withColumn(\n",
    "        \"is_power_user\",\n",
    "        (col(\"rating_count\") >= 500)\n",
    "        & (col(\"distinct_genres_rated\") >= 5)\n",
    "        & (col(\"tag_count\") >= 1)\n",
    "    )\n",
    ")\n",
    "\n",
    "dim_users.write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"gold_dim_users\"))\n",
    "print(f\"gold_dim_users: {dim_users.count():,} rows written.\")\n",
    "print(f\"Power users: {dim_users.filter(col('is_power_user')).count():,}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# ── dim_movies_enriched ───────────────────────────────────────────────────────\n",
    "movie_rating_aggs = (\n",
    "    silver_ratings\n",
    "    .groupBy(\"movieId\")\n",
    "    .agg(\n",
    "        spark_round(avg(\"rating\"), 2).alias(\"avg_rating\"),\n",
    "        count(\"*\").alias(\"rating_count\"),\n",
    "    )\n",
    ")\n",
    "\n",
    "movie_tag_counts = (\n",
    "    silver_tags\n",
    "    .groupBy(\"movieId\")\n",
    "    .agg(count(\"*\").alias(\"tag_count\"))\n",
    ")\n",
    "\n",
    "enrichment_clean = (\n",
    "    bronze_enrichment\n",
    "    .drop(\"_ingestion_timestamp\", \"_source_file\", \"title\", \"tmdbId\")\n",
    "    .withColumn(\"budget\",  when(col(\"budget\")  == 0, lit(None)).otherwise(col(\"budget\")))\n",
    "    .withColumn(\"revenue\", when(col(\"revenue\") == 0, lit(None)).otherwise(col(\"revenue\")))\n",
    ")\n",
    "\n",
    "dim_movies_enriched = (\n",
    "    silver_movies_with_links\n",
    "    .join(movie_rating_aggs, \"movieId\", \"left\")\n",
    "    .join(movie_tag_counts,  \"movieId\", \"left\")\n",
    "    .join(enrichment_clean,  \"movieId\", \"left\")\n",
    "    .fillna(0, subset=[\"rating_count\", \"tag_count\"])\n",
    "    .withColumn(\n",
    "        \"profit\",\n",
    "        when(col(\"revenue\").isNotNull() & col(\"budget\").isNotNull(),\n",
    "             col(\"revenue\") - col(\"budget\"))\n",
    "    )\n",
    "    .withColumn(\"has_enrichment\", col(\"directors\").isNotNull())\n",
    ")\n",
    "\n",
    "dim_movies_enriched.write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"gold_dim_movies_enriched\"))\n",
    "print(f\"gold_dim_movies_enriched: {dim_movies_enriched.count():,} rows written.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# ── Null Audit ────────────────────────────────────────────────────────────────\n",
    "gold_tables = {\n",
    "    \"gold_fact_ratings\":       spark.table(tbl(\"gold_fact_ratings\")),\n",
    "    \"gold_dim_users\":          spark.table(tbl(\"gold_dim_users\")),\n",
    "    \"gold_dim_movies_enriched\":spark.table(tbl(\"gold_dim_movies_enriched\")),\n",
    "}\n",
    "\n",
    "for name, df in gold_tables.items():\n",
    "    print(f\"\\n=== {name} ({df.count():,} rows) ===\")\n",
    "    null_counts = df.select(\n",
    "        [spark_sum(col(c).isNull().cast(\"int\")).alias(c) for c in df.columns]\n",
    "    ).collect()[0]\n",
    "    for c in df.columns:\n",
    "        print(f\"  {c:<30} {null_counts[c]:>10,} nulls\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# ── Highest Rated Director (evaluation query) ─────────────────────────────────\n",
    "dme = spark.table(tbl(\"gold_dim_movies_enriched\"))\n",
    "\n",
    "director_rankings = (\n",
    "    dme\n",
    "    .filter(col(\"has_enrichment\") & col(\"avg_rating\").isNotNull())\n",
    "    .select(explode(\"directors\").alias(\"director\"), \"avg_rating\", \"rating_count\")\n",
    "    .groupBy(\"director\")\n",
    "    .agg(\n",
    "        spark_round(avg(\"avg_rating\"), 2).alias(\"director_avg_rating\"),\n",
    "        count(\"*\").alias(\"movie_count\"),\n",
    "        spark_sum(\"rating_count\").alias(\"total_ratings\"),\n",
    "    )\n",
    "    .filter(col(\"movie_count\") >= 2)\n",
    "    .orderBy(col(\"director_avg_rating\").desc())\n",
    ")\n",
    "\n",
    "print(\"=== Highest Rated Directors (2+ movies in top 500) ===\")\n",
    "director_rankings.show(10, truncate=False)"
   ],
   "outputs": [],
   "execution_count": null
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

### Step 2: Verify expected output shape

```
=== gold_fact_ratings (32,000,204 rows) ===
  userId                              0 nulls
  movieId                             0 nulls
  rating                              0 nulls
  ...

=== Highest Rated Directors ===
Francis Ford Coppola  4.23  ...
Frank Darabont        4.22  ...
```

### Step 3: Commit

```bash
git add databricks/03_transform.ipynb
git commit -m "feat: add 03_transform Databricks notebook (Gold layer)"
```

---

## Task 4: Create `databricks/04_train.ipynb` (ALS Model)

**Files:**
- Create: `databricks/04_train.ipynb`

New notebook — trains ALS, evaluates RMSE, saves model to Volume, generates top-10 recommendations per user, writes `gold_recommendations` Delta table.

### Step 1: Create the notebook file

**Full notebook content:**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 04 — Train (ALS Recommendation Model)\n", "\n", "Trains a Spark MLlib ALS model on `gold_fact_ratings`.\n", "- Evaluates RMSE on a 20% held-out test set\n", "- Saves model to a Unity Catalog Volume\n", "- Generates top-10 movie recommendations per user\n", "- Writes `gold_recommendations` Delta table"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "dbutils.widgets.text(\"catalog\",       \"main\",                        \"Unity Catalog name\")\n",
    "dbutils.widgets.text(\"schema\",        \"movielens\",                   \"Schema / database name\")\n",
    "dbutils.widgets.text(\"volume_path\",   \"/Volumes/main/movielens/data\",\"Volume root path\")\n",
    "dbutils.widgets.text(\"als_rank\",      \"10\",  \"ALS rank (latent factors)\")\n",
    "dbutils.widgets.text(\"als_max_iter\",  \"10\",  \"ALS max iterations\")\n",
    "dbutils.widgets.text(\"als_reg_param\", \"0.1\", \"ALS regularization parameter\")\n",
    "\n",
    "CATALOG   = dbutils.widgets.get(\"catalog\")\n",
    "SCHEMA    = dbutils.widgets.get(\"schema\")\n",
    "VOLUME    = dbutils.widgets.get(\"volume_path\")\n",
    "RANK      = int(dbutils.widgets.get(\"als_rank\"))\n",
    "MAX_ITER  = int(dbutils.widgets.get(\"als_max_iter\"))\n",
    "REG_PARAM = float(dbutils.widgets.get(\"als_reg_param\"))\n",
    "MODEL_PATH = f\"{VOLUME}/models/als_model\"\n",
    "\n",
    "def tbl(name):\n",
    "    return f\"{CATALOG}.{SCHEMA}.{name}\"\n",
    "\n",
    "print(f\"Catalog:    {CATALOG}\")\n",
    "print(f\"Schema:     {SCHEMA}\")\n",
    "print(f\"Model path: {MODEL_PATH}\")\n",
    "print(f\"ALS params: rank={RANK}, maxIter={MAX_ITER}, regParam={REG_PARAM}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "from pyspark.sql.functions import col, explode, round as spark_round\n",
    "from pyspark.sql.types import FloatType\n",
    "from pyspark.ml.recommendation import ALS\n",
    "from pyspark.ml.evaluation import RegressionEvaluator\n",
    "\n",
    "fact_ratings = spark.table(tbl(\"gold_fact_ratings\"))\n",
    "\n",
    "als_data = fact_ratings.select(\n",
    "    col(\"userId\"),\n",
    "    col(\"movieId\"),\n",
    "    col(\"rating\").cast(FloatType()),\n",
    ")\n",
    "\n",
    "n_ratings = als_data.count()\n",
    "n_users   = als_data.select(\"userId\").distinct().count()\n",
    "n_movies  = als_data.select(\"movieId\").distinct().count()\n",
    "sparsity  = 1 - (n_ratings / (n_users * n_movies))\n",
    "\n",
    "print(f\"Users:    {n_users:>12,}\")\n",
    "print(f\"Movies:   {n_movies:>12,}\")\n",
    "print(f\"Ratings:  {n_ratings:>12,}\")\n",
    "print(f\"Sparsity: {sparsity:>12.4%}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "train, test = als_data.randomSplit([0.8, 0.2], seed=42)\n",
    "train.cache()\n",
    "test.cache()\n",
    "print(f\"Train: {train.count():,}  |  Test: {test.count():,}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "als = ALS(\n",
    "    rank=RANK,\n",
    "    maxIter=MAX_ITER,\n",
    "    regParam=REG_PARAM,\n",
    "    userCol=\"userId\",\n",
    "    itemCol=\"movieId\",\n",
    "    ratingCol=\"rating\",\n",
    "    coldStartStrategy=\"drop\",\n",
    "    implicitPrefs=False,\n",
    "    seed=42,\n",
    ")\n",
    "\n",
    "print(\"Training ALS model...\")\n",
    "model = als.fit(train)\n",
    "print(\"Training complete.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "evaluator = RegressionEvaluator(\n",
    "    metricName=\"rmse\",\n",
    "    labelCol=\"rating\",\n",
    "    predictionCol=\"prediction\",\n",
    ")\n",
    "\n",
    "predictions = model.transform(test)\n",
    "rmse = evaluator.evaluate(predictions)\n",
    "print(f\"Test RMSE: {rmse:.4f}\")\n",
    "\n",
    "train.unpersist()\n",
    "test.unpersist()"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "model.write().overwrite().save(MODEL_PATH)\n",
    "print(f\"Model saved to: {MODEL_PATH}\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Generate top-10 recommendations for all users\n",
    "recs_raw = model.recommendForAllUsers(10)\n",
    "\n",
    "# Flatten: (userId, [Row(movieId, rating), ...]) -> (userId, movieId, predicted_rating)\n",
    "recs_flat = (\n",
    "    recs_raw\n",
    "    .select(\"userId\", explode(\"recommendations\").alias(\"rec\"))\n",
    "    .select(\n",
    "        col(\"userId\"),\n",
    "        col(\"rec.movieId\").alias(\"movieId\"),\n",
    "        spark_round(col(\"rec.rating\").cast(FloatType()), 4).alias(\"predicted_rating\"),\n",
    "    )\n",
    ")\n",
    "\n",
    "# Enrich with movie title\n",
    "dim_movies = spark.table(tbl(\"gold_dim_movies_enriched\")).select(\"movieId\", \"title\", \"genres\", \"avg_rating\")\n",
    "gold_recommendations = recs_flat.join(dim_movies, \"movieId\", \"left\")\n",
    "\n",
    "gold_recommendations.write.format(\"delta\").mode(\"overwrite\").saveAsTable(tbl(\"gold_recommendations\"))\n",
    "print(f\"gold_recommendations: {gold_recommendations.count():,} rows written.\")"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Demo: show top-10 recommendations for the first user\n",
    "sample_user = gold_recommendations.select(\"userId\").first()[0]\n",
    "\n",
    "print(f\"=== Top-10 Recommendations for userId={sample_user} ===\")\n",
    "(\n",
    "    spark.table(tbl(\"gold_recommendations\"))\n",
    "    .filter(col(\"userId\") == sample_user)\n",
    "    .orderBy(col(\"predicted_rating\").desc())\n",
    "    .select(\"title\", \"predicted_rating\", \"avg_rating\", \"genres\")\n",
    "    .show(10, truncate=False)\n",
    ")\n",
    "\n",
    "print(f\"\\nModel RMSE: {rmse:.4f}\")\n",
    "print(\"Pipeline complete: Ingest → Clean → Transform → Train ✓\")"
   ],
   "outputs": [],
   "execution_count": null
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

### Step 2: Verify expected output shape

```
Training ALS model...
Training complete.
Test RMSE: ~0.85

gold_recommendations: 2,009,480 rows written.

=== Top-10 Recommendations for userId=... ===
title                predicted_rating  avg_rating  genres
...
```

### Step 3: Commit

```bash
git add databricks/04_train.ipynb
git commit -m "feat: add 04_train Databricks notebook (ALS model)"
```

---

## Summary Table

| Notebook | Layer | Reads | Writes | Key Delta Tables |
|---|---|---|---|---|
| `01_ingest.ipynb` | Bronze | CSVs from Volume + TMDB API | 5 Delta tables | `bronze_ratings`, `bronze_movies`, `bronze_links`, `bronze_tags`, `bronze_enrichment` |
| `02_clean.ipynb` | Silver | Bronze Delta tables | 5 Delta tables | `silver_ratings`, `silver_movies`, `silver_links`, `silver_tags`, `silver_movies_with_links` |
| `03_transform.ipynb` | Gold | Silver Delta + Bronze enrichment | 3 Delta tables | `gold_fact_ratings`, `gold_dim_users`, `gold_dim_movies_enriched` |
| `04_train.ipynb` | ML | Gold Delta | 1 Delta table + model | `gold_recommendations`, ALS model at Volume path |

## Databricks Workflow Setup Notes

After uploading notebooks, create a Workflow with 4 tasks:
1. Task `ingest` → `databricks/01_ingest.ipynb`
2. Task `clean` → `databricks/02_clean.ipynb` (depends on: ingest)
3. Task `transform` → `databricks/03_transform.ipynb` (depends on: clean)
4. Task `train` → `databricks/04_train.ipynb` (depends on: transform)

Set job-level parameters:
- `catalog` = your catalog name
- `schema` = your schema name
- `volume_path` = `/Volumes/<catalog>/<schema>/<volume>`
- `secret_scope` = the scope name where `TMDB_BEARER_TOKEN` is stored

All 4 notebooks read these values from `dbutils.widgets.get()` — no hardcoded paths anywhere.
