# Databricks notebook source
from __future__ import annotations

from typing import List, Tuple

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

# -----------------------------
# Config
# -----------------------------
GOLD = "workspace.movielens_gold"

# Your existing volume
BASE_VOL_DBFS = "dbfs:/Volumes/workspace/movielens/movielens_files"
ARTIFACTS_DBFS = f"{BASE_VOL_DBFS}/artifacts"
MODEL_PATH_DBFS = f"{ARTIFACTS_DBFS}/als_model"

# Serving/demo table
DEMO_RECS_TABLE = f"{GOLD}.als_recommendations_demo_flat"

# Clamp range for displayed/served scores
SCORE_MIN = 0.5
SCORE_MAX = 5.0

# Demo users
DEMO_USERS = [1, 10, 100]

# Candidate pool size for demo scoring
CANDIDATE_TOPK = 5000

# ALS hyperparams
ALS_RANK = 50
ALS_MAXITER = 10
ALS_REGPARAM = 0.1

# -----------------------------
# Read Gold tables
# -----------------------------
train_df = (
    spark.table(f"{GOLD}.train_view_als")
    .select(
        F.col("userId").cast("int").alias("userId"),
        F.col("movieId").cast("int").alias("movieId"),
        F.col("rating").cast("float").alias("rating"),
    )
    .filter(F.col("rating").isNotNull())
    .filter((F.col("rating") >= F.lit(SCORE_MIN)) & (F.col("rating") <= F.lit(SCORE_MAX)))
)

movies_dim = (
    spark.table(f"{GOLD}.dim_movies_enriched")
    .select(F.col("movieId").cast("int").alias("movieId"), F.col("title").cast("string").alias("title"))
)

# -----------------------------
# Preflight: make sure artifacts folder exists
# -----------------------------
print("Preflight: listing base volume")
display(dbutils.fs.ls(BASE_VOL_DBFS))

# Ensure artifacts folder exists
try:
    dbutils.fs.ls(ARTIFACTS_DBFS)
    print("Artifacts folder ok:", ARTIFACTS_DBFS)
except Exception:
    dbutils.fs.mkdirs(ARTIFACTS_DBFS)
    print("Created artifacts folder:", ARTIFACTS_DBFS)

print("Model path:", MODEL_PATH_DBFS)

# -----------------------------
# Split: train/val/test
# -----------------------------
train_split, val_split, test_split = train_df.randomSplit([0.90, 0.05, 0.05], seed=42)

print(
    "Counts:",
    "train", train_split.count(),
    "val", val_split.count(),
    "test", test_split.count(),
)

# -----------------------------
# Train ALS
# -----------------------------
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    nonnegative=True,
    coldStartStrategy="drop",
    rank=ALS_RANK,
    maxIter=ALS_MAXITER,
    regParam=ALS_REGPARAM,
)

model = als.fit(train_split)

# -----------------------------
# Evaluate
# -----------------------------
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

val_pred = model.transform(val_split)
val_rmse = evaluator.evaluate(val_pred)

test_pred = model.transform(test_split)
test_rmse = evaluator.evaluate(test_pred)

print(f"VAL RMSE:  {val_rmse:.4f}")
print(f"TEST RMSE: {test_rmse:.4f}")

# -----------------------------
# Save model to Volume
# -----------------------------
try:
    dbutils.fs.rm(MODEL_PATH_DBFS, recurse=True)
except Exception:
    pass

model.write().overwrite().save(MODEL_PATH_DBFS)
print("Saved model to:", MODEL_PATH_DBFS)

# -----------------------------
# Demo Recommendations WITHOUT recommendForAllUsers()
# -----------------------------

# 1) userFactors & itemFactors
uf = model.userFactors.select(F.col("id").cast("int").alias("userId"), F.col("features").alias("u"))
it = model.itemFactors.select(F.col("id").cast("int").alias("movieId"), F.col("features").alias("v"))

demo_users_df = spark.createDataFrame([(u,) for u in DEMO_USERS], "userId int")
uf_demo = uf.join(demo_users_df, on="userId", how="inner")

# 2) candidate movies = most-rated movies
popularity = train_df.groupBy("movieId").agg(F.count("*").alias("n_ratings"))
candidates = (
    popularity.orderBy(F.desc("n_ratings"))
    .limit(CANDIDATE_TOPK)
    .select("movieId")
    .join(it, on="movieId", how="inner")
)

# 3) remove already-seen movies for demo users
seen = (
    train_df.join(demo_users_df, on="userId", how="inner")
    .select("userId", "movieId")
    .distinct()
)

# 4) score = dot(user_features, item_features) via UDF (no higher-order functions)
@F.udf(T.DoubleType())
def dot(a: List[float], b: List[float]) -> float:
    if a is None or b is None:
        return float("nan")
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        s += float(a[i]) * float(b[i])
    return s

scored = (
    uf_demo.crossJoin(candidates)
    .select("userId", "movieId", dot(F.col("u"), F.col("v")).alias("score_raw"))
    .join(seen, on=["userId", "movieId"], how="left_anti")
)

# 5) clamp scores to [0.5, 5.0]
scored_clamped = (
    scored
    .withColumn(
        "score",
        F.when(F.col("score_raw").isNull(), F.lit(None).cast("double"))
        .when(F.col("score_raw") < F.lit(SCORE_MIN), F.lit(SCORE_MIN))
        .when(F.col("score_raw") > F.lit(SCORE_MAX), F.lit(SCORE_MAX))
        .otherwise(F.col("score_raw"))
    )
)

# 6) Top-N per user
w = Window.partitionBy("userId").orderBy(F.desc("score"), F.desc("score_raw"))

topn = (
    scored_clamped
    .withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") <= F.lit(10))
    .drop("rn")
)

# 7) Enrich titles and write demo table
topn_enriched = (
    topn.join(movies_dim, on="movieId", how="left")
    .select("userId", "movieId", "score", "title", "score_raw")
)

(
    topn_enriched.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(DEMO_RECS_TABLE)
)

print("Saved demo recommendations to:", DEMO_RECS_TABLE)

display(
    spark.table(DEMO_RECS_TABLE)
    .orderBy("userId", F.desc("score"), F.desc("score_raw"))
)
