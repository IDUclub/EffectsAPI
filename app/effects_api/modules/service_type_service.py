import asyncio
import re
from typing import Dict, Iterable, List, Mapping, Optional, cast

import geopandas as gpd
import pandas as pd
from blocksnet.analysis.indicators.socio_economic import SocialIndicator
from blocksnet.config import service_types_config
from loguru import logger

from app.clients.urban_api_client import UrbanAPIClient
from app.common.caching.caching_service import FileCache
from app.common.utils.effects_utils import EffectsUtils
from app.effects_api.constants.const import SERVICE_TYPES_MAPPING

_SOCIAL_VALUES_BY_ST: Dict[int, Optional[List[int]]] = {}
_SOCIAL_VALUES_LOCK = asyncio.Lock()
_SERVICE_NAME_TO_ID: dict[str, int] = {
    name: sid for sid, name in SERVICE_TYPES_MAPPING.items()
}
_VALID_SERVICE_NAMES: set[str] = set(_SERVICE_NAME_TO_ID.keys())
_NUM_SUFFIX_RE = re.compile(r"^\d+$")

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
    df["blocksnet"] = df.apply(lambda s: SERVICE_TYPES_MAPPING.get(s.name), axis=1)

    # return df[["name", "infrastructure_type", "infrastructure_weight", "social_values"]]
    return df


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
            if isinstance(ru_name, pd.Series):
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


def adapt_social_service_types_df(
    service_types_df: pd.DataFrame,
    mapping: Mapping["SocialIndicator", Iterable[int]],
) -> pd.DataFrame:
    """
    Attach 'indicator' column to social service types using SOCIAL_INDICATORS_MAPPING
    and normalize naming convention.

    Parameters
    ----------
    service_types_df : pd.DataFrame
        DataFrame where index represents service_type_id.
    mapping : Mapping[SocialIndicator, list[int]]
        Mapping from SocialIndicator enum to service_type IDs.

    Returns
    -------
    pd.DataFrame
        Adapted DataFrame with added 'indicator' column and renamed columns.
    """
    df = service_types_df.copy()

    id_to_indicator = {
        st_id: indicator for indicator, ids in mapping.items() for st_id in ids
    }

    df["indicator"] = df.index.map(id_to_indicator)

    df = df.rename(
        columns={
            "radius_availability_meters": "meters",
            "time_availability_minutes": "minutes",
            "services_per_1000_normative": "count",
            "services_capacity_per_1000_normative": "capacity",
        }
    )

    return df


def _build_name_maps(
    service_types_df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build lookups from service type names to ids."""
    if service_types_df.index.name is None:
        service_types_df = service_types_df.copy()
        service_types_df.index.name = "service_type_id"

    id_series = (
        pd.to_numeric(service_types_df.index.to_series(), errors="coerce")
        .dropna()
        .astype("int64")
    )

    name_to_id: dict[str, int] = {}
    blocksnet_to_id: dict[str, int] = {}

    for sid, row in service_types_df.loc[id_series.index].iterrows():
        try:
            sid_int = int(sid)
        except Exception:
            continue

        nm = row.get("name")
        if isinstance(nm, str) and nm:
            prev = name_to_id.get(nm)
            if prev is not None and prev != sid_int:
                logger.warning(
                    f"Duplicate mapping for name {nm}: {prev} -> {sid_int} (last wins)",
                    nm,
                    prev,
                    sid_int,
                )
            name_to_id[nm] = sid_int

        bn = row.get("blocksnet")
        if isinstance(bn, str) and bn:
            prev = blocksnet_to_id.get(bn)
            if prev is not None and prev != sid_int:
                logger.warning(
                    f"Duplicate mapping for blocksnet {bn}: {prev} -> {sid_int} (last wins)"
                )
            blocksnet_to_id[bn] = sid_int

    return name_to_id, blocksnet_to_id


def _rename_non_id_columns_to_ids(
    df: pd.DataFrame,
    name_to_id: dict[str, int],
    blocksnet_to_id: dict[str, int],
    prefixes: Iterable[str],
) -> pd.DataFrame:
    """
    Rename columns like 'count_kindergarten' -> 'count_21' using provided lookups.
    Leaves 'count_21' (already id) as-is. Unknown names are kept and warned.
    """
    rename_map: dict[str, str] = {}

    for col in df.columns:
        for prefix in prefixes:
            pref = f"{prefix}_"
            if not col.startswith(pref):
                continue
            suffix = col[len(pref) :]

            suffix = col[len(pref) :]
            if suffix.isnumeric():
                break

            sid = blocksnet_to_id.get(suffix) or name_to_id.get(suffix)
            if sid is not None:
                rename_map[col] = f"{pref}{sid}"
            else:
                logger.warning(f"No service_id mapping found for column '{col}")
            break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def ensure_missing_id_and_name_columns(
    blocks_gdf: gpd.GeoDataFrame,
    count_prefix: str = "count",
    capacity_prefix: str = "capacity",
) -> gpd.GeoDataFrame:
    """
    Ensure per-block columns exist for every service in SERVICE_TYPES_MAPPING:
      - {count_prefix}_{id}
      - {capacity_prefix}_{id}
      - {count_prefix}_{name}
      - {capacity_prefix}_{name}
    Missing columns are created and filled with zeros.

    Parameters
    ----------
    blocks_gdf : GeoDataFrame
        Blocks data.
    service_type_mapping : dict[int -> str or None]
        Mapping of service_type_id to service_name.
    count_prefix, capacity_prefix : str
        Column prefixes.

    Returns
    -------
    GeoDataFrame
        Updated dataframe containing all required columns.
    """
    ids = sorted(int(sid) for sid in SERVICE_TYPES_MAPPING.keys())
    names = [
        name
        for name in SERVICE_TYPES_MAPPING.values()
        if isinstance(name, str) and name.strip()
    ]

    required_cols = []
    for sid in ids:
        required_cols.append(f"{count_prefix}_{sid}")
        required_cols.append(f"{capacity_prefix}_{sid}")
    for name in names:
        required_cols.append(f"{count_prefix}_{name}")
        required_cols.append(f"{capacity_prefix}_{name}")

    missing = [c for c in required_cols if c not in blocks_gdf.columns]
    if missing:
        logger.info(f"Creating missing service columns (zeros): {missing}")
        add = {}
        for col in missing:
            if col.startswith(f"{count_prefix}_"):
                add[col] = pd.Series(0, index=blocks_gdf.index, dtype="int64")
            else:
                add[col] = pd.Series(0.0, index=blocks_gdf.index, dtype="float64")

        blocks_gdf = blocks_gdf.join(pd.DataFrame(add, index=blocks_gdf.index))

    return blocks_gdf


def generate_blocksnet_columns(
    blocks_gdf: gpd.GeoDataFrame,
    service_types_df: pd.DataFrame,
    count_prefix: str = "count",
    capacity_prefix: str = "capacity",
    strict: bool = False,
) -> gpd.GeoDataFrame:
    """
    Build notebook-like aggregated columns by 'blocksnet' groups:

    1) Normalize per-service columns in `blocks_gdf`:
       - Accept either '<prefix>_<service_id>' OR '<prefix>_<name|blocksnet>'.
       - Non-numeric suffixes are renamed to numeric ids using `service_types_df`.

    2) Aggregate by `blocksnet`:
       - sum of count_<service_id> -> count_<blocksnet>
       - sum of capacity_<service_id> -> capacity_<blocksnet>

    Robust behavior:
      - service_type_id taken from index or 'service_type_id' column
      - ids cast to int
      - sum only existing columns; missing -> warning (or raise if strict=True)
    """
    st_df = service_types_df[service_types_df["blocksnet"].notna()].copy()

    if "service_type_id" in st_df.columns:
        ids_series = st_df["service_type_id"]
    else:
        ids_series = st_df.index.to_series(name="service_type_id")

    ids_series = pd.to_numeric(ids_series, errors="coerce").astype("Int64")
    st_df = st_df.assign(service_type_id=ids_series).dropna(subset=["service_type_id"])
    st_df["service_type_id"] = st_df["service_type_id"].astype("int64")

    name_to_id, blocksnet_to_id = _build_name_maps(st_df)
    blocks_gdf = _rename_non_id_columns_to_ids(
        blocks_gdf,
        name_to_id=name_to_id,
        blocksnet_to_id=blocksnet_to_id,
        prefixes=(count_prefix, capacity_prefix),
    )

    grouped = (
        st_df.groupby("blocksnet")["service_type_id"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .to_dict()
    )

    new_columns: dict[str, pd.Series] = {}
    for bnet, st_ids in grouped.items():
        for prefix in (count_prefix, capacity_prefix):
            expected = [f"{prefix}_{sid}" for sid in st_ids]
            existing = [c for c in expected if c in blocks_gdf.columns]
            missing = [c for c in expected if c not in blocks_gdf.columns]

            if missing:
                msg = f"Missing columns for '{bnet}' [{prefix}]: {missing}"
                if strict:
                    raise KeyError(msg)
                logger.warning(msg)

            series = (
                blocks_gdf[existing].fillna(0).sum(axis=1)
                if existing
                else pd.Series(0, index=blocks_gdf.index, dtype="float64")
            )
            new_columns[f"{prefix}_{bnet}"] = series

    out = pd.concat(
        [blocks_gdf, pd.DataFrame(new_columns, index=blocks_gdf.index)], axis=1
    )
    return cast(gpd.GeoDataFrame, out)
