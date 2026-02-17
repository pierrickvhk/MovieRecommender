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
