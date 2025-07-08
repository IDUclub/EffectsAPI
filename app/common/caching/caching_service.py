import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

_CACHE_DIR = Path().absolute() / "__effects_cache__"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe(s: str) -> str:
    return _FILENAME_RE.sub("", s)


def _file_name(method: str, scenario_id: int, ts: datetime) -> Path:
    stamp = ts.strftime("%Y%m%d_%H%M")
    name = f"scenario_{scenario_id}_{_safe(method)}_{stamp}.json"
    return _CACHE_DIR / name


class FileCache:
    """Service for caching files."""

    def save(
        self,
        method: str,
        scenario_id: int,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> Path:
        ts = datetime.now()
        path = _file_name(method, scenario_id, ts)
        if path.exists():
            return path
        to_save = {
            "meta": {"timestamp": ts.isoformat(), "params": params},
            "data": data,
        }
        path.write_text(json.dumps(to_save, ensure_ascii=False))
        return path

    def _latest_path(self, method: str, scenario_id: int) -> Path | None:
        pattern = f"scenario_{scenario_id}_{_safe(method)}_*.json"
        files = sorted(_CACHE_DIR.glob(pattern), reverse=True)
        return files[0] if files else None

    def load(
        self, method: str, scenario_id: int, max_age: timedelta | None = None
    ) -> dict[str, Any] | None:
        path = self._latest_path(method, scenario_id)
        if not path:
            return None
        if max_age:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > max_age:
                return None
        return json.loads(path.read_text())

    def has(
        self, method: str, scenario_id: int, max_age: timedelta | None = None
    ) -> bool:
        return self.load(method, scenario_id, max_age) is not None


cache = FileCache()
