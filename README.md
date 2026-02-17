# MovieRecommender# 🎬 MovieLens Lakehouse Recommender (Databricks + Spark + Delta)

Production-style lakehouse pipeline on MovieLens (raw → bronze → silver → gold) and an ALS recommender that writes top-10 recommendations per user.

## Architecture (Medallion)
- **Raw files**: CSVs uploaded to a Databricks Volume
- **Bronze**: schema-enforced Delta tables (no transformations)
- **Silver**: cleaning, typing, dedup, joins, data-quality metrics
- **Gold**: fact/dims + ML training view
- **Serving**: Spark MLlib ALS → `gold_user_recommendations`

## Repo Structure
- `src/movielens/` — reusable, typed pipeline code (IO, transforms, ML)
- `notebooks/` — orchestration notebooks (Databricks Jobs/Workflows)
- `workflows/` — Databricks job definitions (JSON)
- `tests/` — unit tests for transforms & quality rules

## Quickstart (Local)
```bash
make install
make format lint type test
```

## Quickstart (Databricks)

1. Create schema(s) and a Volume
2. Upload `ratings.csv`, `movies.csv`, `links.csv`, `tags.csv` to the Volume
3. Run notebooks in order:
   1. `01_bronze_ingest.py`
   2. `02_silver_clean_join.py`
   3. `03_gold_features.py`
   4. `04_train_als.py`

## Data Sources

- MovieLens CSV datasets (`ratings` / `movies` / `links` / `tags`)
- Optional enrichment (JSON) added later for director/budget/poster metadata

## Demo Queries

See `notebooks/05_demo_queries.sql`.

## Tradeoffs (v0.1)

- SCD1 overwrite for Silver tables (idempotent + simple)
- Partition strategy documented in Gold
- Model cold-start handled via `coldStartStrategy="drop"`

