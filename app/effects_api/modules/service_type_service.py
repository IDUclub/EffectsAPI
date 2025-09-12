import asyncio
from typing import Dict, List, Optional

import pandas as pd
from blocksnet.config import service_types_config

from app.clients.urban_api_client import UrbanAPIClient
from app.common.caching.caching_service import FileCache
from app.effects_api.constants.const import SERVICE_TYPES_MAPPING

for st_id, st_name in SERVICE_TYPES_MAPPING.items():
    if st_name is None:
        continue
    assert st_name in service_types_config, f"{st_id}:{st_name} not in config"


async def _adapt_name(service_type_id: int) -> Optional[str]:
    return SERVICE_TYPES_MAPPING.get(service_type_id)


_SOCIAL_VALUES_BY_ST: Dict[int, Optional[List[int]]] = {}
_SOCIAL_VALUES_LOCK = asyncio.Lock()


async def _warmup_social_values(
    service_type_ids: List[int], client: UrbanAPIClient
) -> None:
    missing = [sid for sid in service_type_ids if sid not in _SOCIAL_VALUES_BY_ST]
    if not missing:
        return
    async with _SOCIAL_VALUES_LOCK:
        missing = [sid for sid in service_type_ids if sid not in _SOCIAL_VALUES_BY_ST]
        if not missing:
            return
        results = await asyncio.gather(
            *(client.get_service_type_social_values(sid) for sid in missing)
        )
        for sid, df in zip(missing, results):
            _SOCIAL_VALUES_BY_ST[sid] = None if df is None else list(df.index)


async def _adapt_social_values(
    service_type_id: int, client: UrbanAPIClient
) -> Optional[List[int]]:
    await _warmup_social_values([service_type_id], client)
    return _SOCIAL_VALUES_BY_ST.get(service_type_id)


async def adapt_service_types(
    service_types_df: pd.DataFrame, client: UrbanAPIClient
) -> pd.DataFrame:
    df = service_types_df[["infrastructure_type"]].copy()
    df["infrastructure_weight"] = service_types_df["weight_value"]

    service_type_ids: List[int] = df.index.tolist()

    names = await asyncio.gather(*(_adapt_name(st_id) for st_id in service_type_ids))
    df["name"] = names
    df = df.dropna(subset=["name"]).copy()

    await _warmup_social_values(list(df.index), client)
    df["social_values"] = [_SOCIAL_VALUES_BY_ST.get(st_id) for st_id in df.index]

    return df[["name", "infrastructure_type", "infrastructure_weight", "social_values"]]


async def get_services_with_ids_from_layer(
    scenario_id: int,
    method: str,
    cache: FileCache,
) -> dict:
    cached: Optional[dict] = cache.load_latest(method, scenario_id)
    if not cached or "data" not in cached:
        return {"before": [], "after": []}

    data = cached["data"]

    def map_services(names: List[str]):
        result = []
        for name in names:
            matched = [
                {"id": sid, "name": sname}
                for sid, sname in SERVICE_TYPES_MAPPING.items()
                if sname == name
            ]
            if matched:
                result.extend(matched)
            else:
                result.append({"id": None, "name": name})
        return result

    if "before" in data or "after" in data:
        before_names = list(data.get("before", {}).keys())
        after_names = list(data.get("after", {}).keys())
        return {
            "before": map_services(before_names),
            "after": map_services(after_names),
        }

    if "provision" in data:
        prov_names = list(data["provision"].keys())
        return {
            "services": map_services(prov_names)
        }

    return {"before": [], "after": []}

