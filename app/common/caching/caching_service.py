import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_CACHE_DIR = Path().absolute() / "__effects_cache__"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe(s: str) -> str:
    return _FILENAME_RE.sub("", s)


def _file_name(method: str, scenario_id: int, phash: str, day: str) -> Path:
    name = f"{day}__scenario_{scenario_id}__{_safe(method)}__{phash}.json"
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
        scenario_id: int,
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

        path = _file_name(method, scenario_id, phash, day)
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

    def _latest_path(self, method: str, scenario_id: int) -> Path | None:
        pattern = f"*__scenario_{scenario_id}__{_safe(method)}__*.json"
        files = sorted(_CACHE_DIR.glob(pattern), reverse=True)
        return files[0] if files else None

    def load(
        self,
        method: str,
        scenario_id: int,
        params_hash: str,
        max_age: timedelta | None = None,
    ) -> dict[str, Any] | None:

        pattern = f"*__scenario_{scenario_id}__{_safe(method)}__{params_hash}.json"
        files = sorted(_CACHE_DIR.glob(pattern), reverse=True)
        if not files:
            return None

        path = files[0]
        if max_age:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > max_age:
                return None

        return json.loads(path.read_text(encoding="utf-8"))

    def load_latest(self, method: str, scenario_id: int) -> dict[str, Any] | None:
        path = self._latest_path(method, scenario_id)
        if not path:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def has(
        self, method: str, scenario_id: int, max_age: timedelta | None = None
    ) -> bool:
        return self.load(method, scenario_id, max_age) is not None

    def parse_task_id(self, task_id: str):
        parts = task_id.split("_")
        if len(parts) < 3:
            return None, None, None

        tail = "_".join(parts[-1:])
        scenario_id_raw = parts[-2]
        method = "_".join(parts[:-2])

        if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail.lower()):
            phash = tail
        else:
            phash = self.params_hash(tail)

        scenario_id = int(scenario_id_raw) if scenario_id_raw.isdigit() else scenario_id_raw
        return method, scenario_id, phash