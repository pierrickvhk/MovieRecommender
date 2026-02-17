from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectNamespaces:
    catalog: str = "workspace"
    schema_raw: str = "movielens"
    schema_bronze: str = "movielens_bronze"
    schema_silver: str = "movielens_silver"
    schema_gold: str = "movielens_gold"

    volume_name: str = "movielens_files"

    @property
    def volume_path(self) -> str:
        # Databricks Volumes path convention
        return f"/Volumes/{self.catalog}/{self.schema_raw}/{self.volume_name}"
