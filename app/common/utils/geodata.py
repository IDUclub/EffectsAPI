import asyncio
import json

import geopandas as gpd
import pandas as pd
from blocksnet.relations import calculate_distance_matrix
from loguru import logger
from shapely.geometry.base import BaseGeometry
from shapely.wkt import dumps, loads

from app.common.exceptions.http_exception_wrapper import http_exception
from app.effects_api.constants.const import COL_RU, SPEED, ROADS_ID
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
    gdf_copy.geometry = await asyncio.to_thread(
        round_coords, gdf_copy.geometry, ndigits
    )

    return json.loads(gdf_copy.to_json(drop_id=True))

def safe_gdf_to_geojson(
    gdf: gpd.GeoDataFrame,
    to_epsg: int = 4326,
    round_ndigits: int = 6,
    drop_cols: tuple[str, ...] = (),
) -> dict:
    """Project, round, sanitize and serialize GeoDataFrame to GeoJSON.

    Steps:
    - Drop unwanted columns (e.g., non-serializable).
    - Project to EPSG (default 4326).
    - Round geometry coordinates to given precision.
    - Ensure all properties are JSON-serializable.
    - Return parsed dict (FeatureCollection).
    """
    logger.info(f"Serializing GeoDataFrame to GeoJSON (EPSG:{to_epsg}, round={round_ndigits})")
    gdf2 = gdf.drop(columns=[c for c in drop_cols if c in gdf.columns]).copy()
    gdf2 = gdf2.to_crs(to_epsg)
    gdf2.geometry = round_coords(gdf2.geometry, round_ndigits)
    return json.loads(gdf2.to_json(drop_id=True))


def fc_to_gdf(fc: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")


def is_fc(obj: dict) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "FeatureCollection" and "features" in obj


def round_coords(
    geometry: gpd.GeoSeries | BaseGeometry, ndigits: int = 6
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

def gdf_join_on_block_id(left: gpd.GeoDataFrame, right: pd.DataFrame, how: str = "left") -> gpd.GeoDataFrame:
    """Join two frames by block_id index safely.

    - Ensures both indices are int.
    - Keeps geometry from the left GeoDataFrame.
    """
    gdf = left.copy()
    gdf.index = gdf.index.astype(int)
    r = right.copy()
    r.index = r.index.astype(int)
    return gdf.join(r, how=how)


def _ensure_block_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure index is integer 'block_id'."""
    if "block_id" in gdf.columns:
        gdf = gdf.copy()
        gdf["block_id"] = gdf["block_id"].astype(int)
        if gdf.index.name == "block_id":
            gdf = gdf.reset_index(drop=True)
        gdf = (
            gdf.drop_duplicates(subset="block_id", keep="last")
            .set_index("block_id")
            .sort_index()
        )
    else:
        gdf = gdf.copy()
        gdf.index = gdf.index.astype(int)
        gdf = gdf[~gdf.index.duplicated(keep="last")].sort_index()
    gdf.index.name = "block_id"
    return gdf

def get_accessibility_matrix(blocks : gpd.GeoDataFrame) -> pd.DataFrame:
  crs = blocks.estimate_utm_crs()
  dist_mx = calculate_distance_matrix(blocks.to_crs(crs))
  return dist_mx // SPEED


async def _roads_overlay_fast(self, scenario_id: int, token: str, target_crs, bounds):
    """
    Try to clip by bbox first, then lighter op than overlay. Offload to executor.
    """
    logger.info(f"Fetching roads for scenario_id={scenario_id}")
    roads_gdf = await self.urban_api_client.get_physical_objects_scenario(
        scenario_id, token=token, physical_object_function_id=ROADS_ID
    )
    if roads_gdf.crs != target_crs:
        roads_gdf = roads_gdf.to_crs(target_crs)

    minx, miny, maxx, maxy = bounds
    roads_gdf = roads_gdf.cx[minx:maxx, miny:maxy]

    def _clip():
        try:
            return roads_gdf
        except Exception:
            return roads_gdf

    return await self._to_executor(_clip)