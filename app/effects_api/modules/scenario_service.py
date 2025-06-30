import shapely
import numpy as np
from blocksnet.preprocessing.imputing import impute_services
from blocksnet.preprocessing.imputing import impute_buildings
from blocksnet.blocks.cutting import preprocess_urban_objects, cut_urban_blocks
import geopandas as gpd
import pandas as pd
from loguru import logger

from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import urban_api_gateway
from app.effects_api.modules.buildings_service import adapt_buildings
from app.effects_api.modules.functional_sources_service import adapt_functional_zones
from app.effects_api.modules.services_service import adapt_services

SOURCES_PRIORITY = ['PZZ', 'OSM', "User"]


def close_gaps(gdf, tolerance):  # taken from momepy
    geom = gdf.geometry.array
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    buffered = shapely.buffer(points, tolerance / 2)

    dissolved = shapely.union_all(buffered)

    exploded = [
        shapely.get_geometry(dissolved, i)
        for i in range(shapely.get_num_geometries(dissolved))
    ]

    centroids = shapely.centroid(exploded)

    snapped = shapely.snap(geom, shapely.union_all(centroids), tolerance)

    return gpd.GeoSeries(snapped, crs=gdf.crs)


async def _get_project_boundaries(project_id: int):
    return gpd.GeoDataFrame(geometry=[await urban_api_gateway.get_project_geometry(project_id)], crs=4326)


async def _get_scenario_roads(scenario_id: int, token: str):
    gdf = await urban_api_gateway.get_physical_objects_scenario(scenario_id, token, physical_object_function_id=26)
    return gdf[['geometry']].reset_index(drop=True)


async def _get_scenario_water(scenario_id: int, token: str):
    gdf = await urban_api_gateway.get_physical_objects_scenario(scenario_id, token, physical_object_function_id=4)
    return gdf[['geometry']].reset_index(drop=True)


async def _get_scenario_blocks(user_scenario_id: int, base_scenario_id: int,
                         boundaries: gpd.GeoDataFrame, token) -> gpd.GeoDataFrame:
    crs = boundaries.crs
    boundaries.geometry = boundaries.buffer(-1)

    water = await _get_scenario_water(user_scenario_id, token)
    water = water.to_crs(crs)
    user_roads = await _get_scenario_roads(user_scenario_id, token)
    user_roads = user_roads.to_crs(crs)
    base_roads =await _get_scenario_roads(base_scenario_id, token)
    base_roads = base_roads.to_crs(crs)
    roads = pd.concat([user_roads, base_roads]).reset_index(drop=True)
    roads.geometry = close_gaps(roads, 1)

    lines, polygons = preprocess_urban_objects(roads, None, water)
    blocks = cut_urban_blocks(boundaries, lines, polygons)
    return blocks


async def _get_scenario_info(scenario_id: int, token: str) -> tuple[int, int]:
    scenario = await urban_api_gateway.get_scenario(scenario_id, token)
    project_id = scenario['project']['project_id']
    project = await urban_api_gateway.get_project(project_id)
    base_scenario_id = project['base_scenario']['id']
    return project_id, base_scenario_id


async def _get_best_functional_zones_source(
        sources_df: pd.DataFrame,
        source: str | None = None,
        year: int | None = None,
) -> tuple[int | None, str | None]:
    """
    Pick the (year, source) pair that should be fetched.

    Rules
    -----
    1. Nothing is given: latest year of the highest-priority source (PZZ).
    2. Only `source`: latest year available for that source.
    3. Only `year`: try that year for priority = PZZ, OSM, User.
    4. 'Both given': use them as-is if a match exists, otherwise fall back to rule 3.
    """
    if source and year:
        row = sources_df.query("source == @source and year == @year")
        if not row.empty:
            return year, source
        year = int(year)
        source = None

    if source and year is None:
        rows = sources_df.query("source == @source")
        if not rows.empty:
            return int(rows["year"].max()), source
        source = None

    if year is not None and source is None:
        for s in SOURCES_PRIORITY:
            row = sources_df.query("source == @s and year == @year")
            if not row.empty:
                return year, s

    for s in SOURCES_PRIORITY:
        rows = sources_df.query("source == @s")
        if not rows.empty:
            return int(rows["year"].max()), s


async def get_scenario_blocks(user_scenario_id: int, token: str) -> gpd.GeoDataFrame:
    project_id, base_scenario_id = await _get_scenario_info(user_scenario_id, token)
    project_boundaries = await _get_project_boundaries(project_id)

    crs = project_boundaries.estimate_utm_crs()
    project_boundaries = project_boundaries.to_crs(crs)

    return await _get_scenario_blocks(user_scenario_id, base_scenario_id, project_boundaries, token)


async def get_scenario_functional_zones(scenario_id: int, token: str, source: str = None, year: int = None) -> gpd.GeoDataFrame:
    source, year = await urban_api_gateway.get_optimal_func_zone_request_data(token, scenario_id, year, source)
    functional_zones = await urban_api_gateway.get_functional_zones_scenario(scenario_id, token, year, source)
    return adapt_functional_zones(functional_zones)


async def get_scenario_buildings(scenario_id: int, token: str):
    try:
        gdf = await urban_api_gateway.get_physical_objects_scenario(scenario_id, token, physical_object_type_id=4, centers_only=True)
        if gdf is None:
            return None
        gdf = adapt_buildings(gdf.reset_index(drop=True))
        crs = gdf.estimate_utm_crs()
        return impute_buildings(gdf.to_crs(crs)).to_crs(4326)
    except Exception as e:
        logger.exception(e)
        raise http_exception(
            404,
            f'No buildings found for scenario {scenario_id}',
            _input={"scenario_id": scenario_id},
            _detail={"error": repr(e)}
        ) from e


async def get_scenario_services(scenario_id: int, service_types: pd.DataFrame, token: str):
    try:
        gdf = await urban_api_gateway.get_services_scenario(scenario_id, centers_only=True, token=token)
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
        gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
        return {st: impute_services(gdf, st) for st, gdf in gdfs.items()}
    except Exception as e:
        logger.error("No buildings found for scenario", e)
