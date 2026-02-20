from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import requests


@dataclass(frozen=True)
class TMDBConfig:
    read_token: str
    base_url: str = "https://api.themoviedb.org/3"
    timeout_s: int = 20
    max_retries: int = 5
    sleep_s: float = 0.25  # polite rate limiting


class TMDBClient:
    def __init__(self, cfg: TMDBConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {cfg.read_token}",
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.cfg.base_url}{path}"
        last_err: Optional[Exception] = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout_s)

                # Handle rate limiting
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(2.0 * attempt, 10.0)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except Exception as e:  # requests exceptions + JSON errors
                last_err = e
                time.sleep(min(0.5 * attempt, 5.0))

        raise RuntimeError(f"TMDB request failed after retries: {url}") from last_err

    def configuration(self) -> Dict[str, Any]:
        # Image URL building uses base_url + size + file_path. :contentReference[oaicite:2]{index=2}
        return self._get("/configuration")

    def movie_details(self, tmdb_id: str) -> Dict[str, Any]:
        return self._get(f"/movie/{tmdb_id}")

    def movie_credits(self, tmdb_id: str) -> Dict[str, Any]:
        # Credits endpoint returns crew incl. Director. :contentReference[oaicite:3]{index=3}
        return self._get(f"/movie/{tmdb_id}/credits")


def pick_director(credits_payload: Dict[str, Any]) -> Optional[str]:
    crew = credits_payload.get("crew") or []
    for member in crew:
        if (member.get("job") == "Director") and member.get("name"):
            return str(member["name"])
    return None


def build_poster_url(cfg_payload: Dict[str, Any], poster_path: Optional[str]) -> Optional[str]:
    if not poster_path:
        return None

    images = cfg_payload.get("images") or {}
    base_url = images.get("secure_base_url") or images.get("base_url")
    sizes = images.get("poster_sizes") or []

    if not base_url or not sizes:
        # Fallback that often works, but config is preferred. :contentReference[oaicite:4]{index=4}
        return f"https://image.tmdb.org/t/p/w500{poster_path}"

    # Choose a reasonable size (prefer w500 if present)
    size = "w500" if "w500" in sizes else sizes[-1]
    return f"{base_url}{size}{poster_path}"


def iter_top_movies_from_spark(limit: int = 500) -> Iterable[Tuple[int, str]]:
    """
    Returns (movieId, tmdbId) for top movies by rating count.
    Assumes Databricks/Spark context: silver_ratings + silver_links exist.
    """
    # Spark is only available when run inside Databricks
    from pyspark.sql import functions as F  # type: ignore

    ratings = spark.table("workspace.movielens_silver.silver_ratings")  # noqa: F821
    links = spark.table("workspace.movielens_silver.silver_links")  # noqa: F821

    top = (
        ratings.groupBy("movieId")
        .agg(F.count("*").alias("n_ratings"))
        .join(links.select("movieId", "tmdbId"), on="movieId", how="inner")
        .filter(F.col("tmdbId").isNotNull())
        .orderBy(F.col("n_ratings").desc())
        .limit(limit)
        .select("movieId", "tmdbId")
        .collect()
    )
    for r in top:
        yield int(r["movieId"]), str(r["tmdbId"])


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    token = os.getenv("TMDB_READ_TOKEN")
    if not token:
        raise SystemExit("Missing env var TMDB_READ_TOKEN (do not hardcode tokens).")

    cfg = TMDBConfig(read_token=token)
    client = TMDBClient(cfg)

    cfg_payload = client.configuration()

    out_path = "/Volumes/workspace/movielens/movielens_files/scraped_metadata.jsonl"

    def gen() -> Iterable[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        for movie_id, tmdb_id in iter_top_movies_from_spark(limit=500):
            details = client.movie_details(tmdb_id)
            credits = client.movie_credits(tmdb_id)

            rec = {
                "movieId": movie_id,
                "tmdbId": tmdb_id,
                "director": pick_director(credits),
                "budget": details.get("budget"),
                "poster_url": build_poster_url(cfg_payload, details.get("poster_path")),
                "source": "tmdb",
                "ingested_at": now,
            }
            yield rec
            time.sleep(cfg.sleep_s)

    n = write_jsonl(out_path, gen())
    print(f"Wrote {n} records to {out_path}")


if __name__ == "__main__":
    main()
