import asyncio
import json

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads, dumps

from app.common.exceptions.http_exception_wrapper import http_exception
from app.effects_api.constants.const import COL_RU
from app.effects_api.modules.scenario_service import SOURCES_PRIORITY


async def gdf_to_ru_fc_rounded(gdf: gpd.GeoDataFrame, ndigits: int = 6) -> dict:
    if "provision_weak" in gdf.columns:
        gdf = gdf.drop(columns="provision_weak")
    gdf = gdf.rename(
        columns={k: v for k, v in COL_RU.items() if k in gdf.columns},
        errors="raise",
    )
    gdf = gdf.to_crs(4326)

    gdf_copy = gdf.copy()
    gdf_copy.geometry = await asyncio.to_thread(round_coords, gdf_copy.geometry, ndigits)

    return json.loads(gdf_copy.to_json(drop_id=True))


def fc_to_gdf(fc: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")

def round_coords(
    geometry: gpd.GeoSeries | BaseGeometry,
    ndigits: int = 6
) -> gpd.GeoSeries | BaseGeometry:
    if isinstance(geometry, gpd.GeoSeries):
        return geometry.map(lambda geom: loads(dumps(geom, rounding_precision=ndigits)))
    elif isinstance(geometry, BaseGeometry):
        return loads(dumps(geometry, rounding_precision=ndigits))
    else:
        raise TypeError("geometry must be GeoSeries or Shapely geometry")


async def get_best_functional_zones_source(
    sources_df: pd.DataFrame,
    source: str | None = None,
    year: int | None = None,
) -> tuple[int | None, str | None]:

    if source and year:
        row = sources_df.query("source == @source and year == @year")
        if not row.empty:
            return year, source
        return await get_best_functional_zones_source(sources_df, None, year)
    elif source and not year:
        rows = sources_df.query("source == @source")
        if not rows.empty:
            return int(rows["year"].max()), source
        return await get_best_functional_zones_source(sources_df, None, year)
    elif year and not source:
        for s in SOURCES_PRIORITY:
            row = sources_df.query("source == @s and year == @year")
            if not row.empty:
                return year, s
    for s in SOURCES_PRIORITY:
        rows = sources_df.query("source == @s")
        if not rows.empty:
            return int(rows["year"].max()), s

    raise http_exception(404, "No available functional zone sources to choose from")
