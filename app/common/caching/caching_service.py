import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from loguru import logger

_CACHE_DIR = Path().absolute() / "__effects_cache__"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe(s: str) -> str:
    return _FILENAME_RE.sub("", s)


PROJECT_BASED_METHODS: set[str] = {
    "social_economical_metrics",
    "urbanomy_metrics",
}


def _owner_prefix(method: str) -> str:
    """Return cache key prefix based on method semantics."""
    return "project" if method in PROJECT_BASED_METHODS else "scenario"


def _file_name(method: str, owner_id: int, phash: str, day: str) -> Path:
    prefix = _owner_prefix(method)
    name = f"{day}__{prefix}_{owner_id}__{_safe(method)}__{phash}.json"
    return _CACHE_DIR / name


def _to_dt(dt_str: str) -> datetime:
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


class FileCache:
    """Service for caching files."""

    def params_hash(self, params: dict[str, Any]) -> str:
        """
        8-symbol md5-hash from params dict.
        """
        raw = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    def save(
        self,
        method: str,
        owner_id: int,
        params: dict[str, Any],
        data: dict[str, Any],
        scenario_updated_at: str | None = None,
    ) -> Path:
        """
        Always write (or overwrite) the cache file so that both
        'before' and 'after' can be stored in the same JSON.
        """
        phash = self.params_hash(params)
        day = datetime.now().strftime("%Y%m%d")

        path = _file_name(method, owner_id, phash, day)
        to_save = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "scenario_updated_at": scenario_updated_at,
                "params_hash": phash,
            },
            "data": data,
        }
        path.write_text(json.dumps(to_save, ensure_ascii=False), encoding="utf-8")
        return path

    def _latest_path(self, method: str, owner_id: int) -> Path | None:
        prefix = _owner_prefix(method)
        pattern = f"*__{prefix}_{owner_id}__{_safe(method)}__*.json"
        files = sorted(_CACHE_DIR.glob(pattern), reverse=True)
        return files[0] if files else None

    def load(
        self,
        method: str,
        owner_id: int,
        params_hash: str,
        max_age: timedelta | None = None,
    ) -> dict[str, Any] | None:
        prefix = _owner_prefix(method)
        pattern = f"*__{prefix}_{owner_id}__{_safe(method)}__{params_hash}.json"
        files = sorted(_CACHE_DIR.glob(pattern), reverse=True)
        if not files:
            return None

        path = files[0]
        if max_age:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > max_age:
                return None

        return json.loads(path.read_text(encoding="utf-8"))

    def load_latest(self, method: str, owner_id: int) -> dict[str, Any] | None:
        path = self._latest_path(method, owner_id)
        if not path:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def has(
        self,
        method: str,
        owner_id: int,
        params_hash: str,
        max_age: timedelta | None = None,
    ) -> bool:
        return self.load(method, owner_id, params_hash, max_age=max_age) is not None

    def parse_task_id(self, task_id: str):
        parts = task_id.split("_")
        if len(parts) < 3:
            return None, None, None

        tail = parts[-1]
        scenario_id_raw = parts[-2]
        method = "_".join(parts[:-2])

        if len(tail) == 8 and tail.lower().strip("0123456789abcdef") == "":
            phash = tail
        else:
            phash = self.params_hash(tail)

        scenario_id = (
            int(scenario_id_raw) if scenario_id_raw.isdigit() else scenario_id_raw
        )
        return method, scenario_id, phash

    def _artifact_path(
        self,
        method: str,
        owner_id: int,
        phash: str,
        name: str,
        ext: Literal["parquet", "pkl"],
    ) -> Path:
        """Build path for a heavy artifact near JSON cache directory."""
        fname = f"artifact__{_safe(method)}__{owner_id}__{phash}__{_safe(name)}.{ext}"
        return _CACHE_DIR / fname

    def save_df_artifact(
        self,
        df: pd.DataFrame,
        method: str,
        owner_id: int,
        params: dict[str, Any],
        name: str,
        fmt: Literal["parquet", "pkl"] = "parquet",
    ) -> Path:
        """
        Save a pandas DataFrame as a heavy artifact.
        fmt='parquet' (default) is compact and fast; fmt='pkl' as a fallback.
        """
        phash = self.params_hash(params)
        path = self._artifact_path(
            method, owner_id, phash, name, "parquet" if fmt == "parquet" else "pkl"
        )

        if fmt == "parquet":
            df.to_parquet(path, index=True)
        else:
            df.to_pickle(path)

        return path

    def load_df_artifact(self, path: Path) -> pd.DataFrame:
        """Load a pandas DataFrame artifact by file extension."""
        ext = path.suffix.lower()
        if ext == ".parquet":
            return pd.read_parquet(path)
        elif ext == ".pkl":
            return pd.read_pickle(path)
        raise ValueError(f"Unsupported artifact extension: {ext}")

    def save_gdf_artifact(
        self,
        gdf: gpd.GeoDataFrame,
        method: str,
        owner_id: int,
        params: dict[str, Any],
        name: str,
        fmt: Literal["parquet", "pkl"] = "parquet",
    ) -> Path:
        phash = self.params_hash(params)
        ext = "parquet" if fmt == "parquet" else "pkl"
        path = self._artifact_path(method, owner_id, phash, name, ext)

        if fmt == "parquet":
            gdf.to_parquet(path, index=True)
        else:
            gdf.to_pickle(path)

        return path

    def load_gdf_artifact(self, path: Path) -> "gpd.GeoDataFrame":
        """Load a GeoDataFrame artifact by file extension."""
        ext = path.suffix.lower()
        if ext == ".parquet":
            return gpd.read_parquet(path)
        elif ext == ".pkl":
            return pd.read_pickle(path)
        raise ValueError(f"Unsupported artifact extension: {ext}")

    def delete_all(self, method: str, owner_id: int) -> int:
        """
        Delete all cached JSON files and heavy artifacts for given method and owner_id.

        Returns:
            Number of deleted files.
        """
        prefix = _owner_prefix(method)

        json_pattern = f"*__{prefix}_{owner_id}__{_safe(method)}__*.json"
        json_files = list(_CACHE_DIR.glob(json_pattern))

        artifact_pattern = f"artifact__{_safe(method)}__{owner_id}__*"
        artifact_files = list(_CACHE_DIR.glob(artifact_pattern))

        deleted = 0
        for path in json_files + artifact_files:
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except Exception as e:
                logger.warning(
                    f"Failed to delete cache file: path={path.as_posix()} err={e}"
                )

        logger.info(
            f"Cache invalidated: method={method} owner_id={owner_id} deleted_files={deleted}"
        )
        return deleted
