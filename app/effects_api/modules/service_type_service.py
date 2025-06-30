import asyncio
from typing import List, Optional

import pandas as pd
from blocksnet.config import service_types_config

from app.dependencies import urban_api_gateway
from app.effects_api.constants.const import SERVICE_TYPES_MAPPING

for st_id, st_name in SERVICE_TYPES_MAPPING.items():
    if st_name is None:
        continue
    assert st_name in service_types_config, f"{st_id}:{st_name} not in config"


async def _adapt_name(service_type_id: int):
    return SERVICE_TYPES_MAPPING.get(service_type_id)


async def _adapt_social_values(service_type_id: int):
    social_values = await urban_api_gateway.get_service_type_social_values(
        service_type_id
    )
    if social_values is None:
        return None
    else:
        return list(social_values.index)


async def adapt_service_types(service_types_df: pd.DataFrame) -> pd.DataFrame:
    df = service_types_df[["infrastructure_type"]].copy()
    df["infrastructure_weight"] = service_types_df["weight_value"]

    service_type_ids = df.index.tolist()

    names: List[Optional[str]] = await asyncio.gather(
        *(_adapt_name(st_id) for st_id in service_type_ids)
    )
    df["name"] = names

    df = df.dropna(subset=["name"]).copy()

    social_vals: List[Optional[List[int]]] = await asyncio.gather(
        *(_adapt_social_values(st_id) for st_id in df.index)
    )
    df["social_values"] = social_vals

    return df[["name", "infrastructure_type", "infrastructure_weight", "social_values"]]
