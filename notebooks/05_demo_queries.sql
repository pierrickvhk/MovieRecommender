# Databricks notebook source
%sql
SHOW TABLES IN workspace.movielens_gold;

-- 1) Power users
SELECT userId, n_ratings, avg_rating, rating_std, is_power_user
FROM workspace.movielens_gold.dim_users
ORDER BY n_ratings DESC
LIMIT 20;

-- 2) Most tagged movies
SELECT m.title, t.n_tags
FROM workspace.movielens_gold.agg_movie_tags t
JOIN workspace.movielens_gold.dim_movies_enriched m
 ON t.movieId = m.movieId
ORDER BY t.n_tags DESC
LIMIT 20;

-- 3) Recommendations for a specific user (demo) 
SELECT userId, score, score_raw, movieId, title
FROM workspace.movielens_gold.als_recommendations_demo_flat
WHERE userId = 1
ORDER BY score DESC
LIMIT 10;

-- 3b) Recommendations for multiple users (1,10,100)
SELECT userId, score, score_raw, movieId, title
FROM workspace.movielens_gold.als_recommendations_demo_flat
WHERE userId IN (1, 10, 100)
ORDER BY userId, score DESC
LIMIT 50;

-- 4) Genre filter example (Spark SQL array contains)
SELECT title, genres
FROM workspace.movielens_gold.dim_movies_enriched
WHERE array_contains(genres, 'Action')
LIMIT 20;

-- WHERE CONCAT('|', genres, '|') LIKE '%|Action|%'

-- 5) Data integrity check: null title in recs (should be 0 ideally)
SELECT COUNT(*) AS recs_with_null_title
FROM workspace.movielens_gold.als_recommendations_demo_flat
WHERE title IS NULL;

-- 6) Sanity check: score range
SELECT MIN(score) AS min_score, MAX(score) AS max_score
FROM workspace.movielens_gold.als_recommendations_demo_flat;

-- 7) Score distribution quick check
SELECT
 percentile_approx(score, 0.01) AS p01,
 percentile_approx(score, 0.50) AS p50,
 percentile_approx(score, 0.90) AS p90,
 percentile_approx(score, 0.99) AS p99
FROM workspace.movielens_gold.als_recommendations_demo_flat;

-- 8) Sanity check: no already-seen movies in recommendations
SELECT COUNT(*) AS recs_already_seen
FROM workspace.movielens_gold.als_recommendations_demo_flat r
JOIN workspace.movielens_gold.fact_ratings fr
 ON r.userId = fr.userId AND r.movieId = fr.movieId;
