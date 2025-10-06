import asyncio
from typing import Dict, List, Optional

import pandas as pd
from blocksnet.config import service_types_config

from app.clients.urban_api_client import UrbanAPIClient
from app.common.caching.caching_service import FileCache
from app.common.utils.ids_convertation import EffectsUtils
from app.effects_api.constants.const import SERVICE_TYPES_MAPPING

_SOCIAL_VALUES_BY_ST: Dict[int, Optional[List[int]]] = {}
_SOCIAL_VALUES_LOCK = asyncio.Lock()
_SERVICE_NAME_TO_ID: dict[str, int] = {
    name: sid for sid, name in SERVICE_TYPES_MAPPING.items()
}
_VALID_SERVICE_NAMES: set[str] = set(_SERVICE_NAME_TO_ID.keys())

for st_id, st_name in SERVICE_TYPES_MAPPING.items():
    if st_name is None:
        continue
    assert st_name in service_types_config, f"{st_id}:{st_name} not in config"


async def _adapt_name(service_type_id: int) -> Optional[str]:
    return SERVICE_TYPES_MAPPING.get(service_type_id)


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


def _map_services(names: list[str]) -> list[dict]:
    out = []
    get_id = _SERVICE_NAME_TO_ID.get
    for n in names:
        sid = get_id(n)
        out.append({"id": sid, "name": n})
    return out


def _filter_service_keys(d: dict | None) -> list[str]:
    if not isinstance(d, dict):
        return []
    return [k for k in d.keys() if k in _VALID_SERVICE_NAMES]


async def get_services_with_ids_from_layer(
    scenario_id: int,
    method: str,
    cache: FileCache,
    utils: EffectsUtils,
    token: str | None = None,
) -> dict:
    if method == "values_oriented_requirements":
        scenario_id = await utils.resolve_base_id(token, scenario_id)

    cached: dict | None = cache.load_latest(method, scenario_id)
    if not cached or "data" not in cached:
        return {"before": [], "after": []}

    data: dict = cached["data"]

    if "before" in data or "after" in data:
        before_names = _filter_service_keys(data.get("before"))
        after_names = _filter_service_keys(data.get("after"))
        return {
            "before": _map_services(before_names),
            "after": _map_services(after_names),
        }

    if "provision" in data:
        prov_names = _filter_service_keys(data["provision"])
        return {"services": _map_services(prov_names)}

    return {"before": [], "after": []}


async def build_en_to_ru_map(service_types_df: pd.DataFrame) -> dict[str, str]:
    russian_names_dict = {}
    for st_id, en_key in SERVICE_TYPES_MAPPING.items():
        if not en_key:
            continue
        if st_id in service_types_df.index:
            ru_name = service_types_df.loc[st_id, "name"]
            if isinstance(ru_name, pd.Series):  # на всякий
                ru_name = ru_name.iloc[0]
            if isinstance(ru_name, str) and ru_name.strip():
                russian_names_dict[en_key] = ru_name
    return russian_names_dict


async def remap_properties_keys_in_geojson(
    geojson: dict, en2ru: dict[str, str]
) -> dict:
    feats = geojson.get("features", [])
    for f in feats:
        props = f.get("properties", {})
        to_rename = [(k, en2ru[k]) for k in props.keys() if k in en2ru]
        for old_k, new_k in to_rename:
            if (
                new_k in props
                and isinstance(props[new_k], dict)
                and isinstance(props[old_k], dict)
            ):
                merged = {**props[old_k], **props[new_k]}
                props[new_k] = merged
            else:
                props[new_k] = props[old_k]
            del props[old_k]
    return geojson
