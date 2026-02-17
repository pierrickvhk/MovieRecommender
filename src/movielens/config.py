from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BronzeTables:
    ratings: str = "bronze_ratings"
    movies: str = "bronze_movies"
    links: str = "bronze_links"
    scraped: str = "bronze_scraped"


@dataclass(frozen=True)
class Paths:
    # Zet deze in je notebook via widgets of environment
    data_root: str  # bv. "/Volumes/main/movielens" of "dbfs:/FileStore/movielens"
    ratings_csv: str = "ratings.csv"
    movies_csv: str = "movies.csv"
    links_csv: str = "links.csv"
    scraped_json: str = "scraped_metadata.jsonl"  # json-lines aanbevolen

    def ratings_path(self) -> str:
        return f"{self.data_root.rstrip('/')}/{self.ratings_csv}"

    def movies_path(self) -> str:
        return f"{self.data_root.rstrip('/')}/{self.movies_csv}"

    def links_path(self) -> str:
        return f"{self.data_root.rstrip('/')}/{self.links_csv}"

    def scraped_path(self) -> str:
        return f"{self.data_root.rstrip('/')}/{self.scraped_json}"
