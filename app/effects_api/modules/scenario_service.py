"""Все методы, которые вызывают ручки по сценарию
А также методы, которые собиракют геослои по сценарию"""
import shapely
import numpy as np
from blocksnet.preprocessing.imputing import impute_services
from blocksnet.preprocessing.imputing import impute_buildings
from blocksnet.blocks.cutting import preprocess_urban_objects, cut_urban_blocks
import geopandas as gpd
import pandas as pd

from app.common.exceptions.http_exception_wrapper import http_exception
from app.effects_api.modules.buildings_service import adapt_buildings
from app.effects_api.modules.functional_sources_service import adapt_functional_zones
from app.effects_api.modules.services_service import adapt_services
from app.effects_api.modules.urban_api_gateway import UrbanAPIGateway

SOURCES_PRIORITY = ['User', 'PZZ', 'OSM']


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
    return gpd.GeoDataFrame(geometry=[await UrbanAPIGateway.get_project_geometry(project_id)], crs=4326)


async def _get_scenario_roads(scenario_id: int):
    gdf = await UrbanAPIGateway.get_physical_objects_scenario(scenario_id, physical_object_function_id=26)
    return gdf[['geometry']].reset_index(drop=True)


async def _get_scenario_water(scenario_id: int):
    gdf = await UrbanAPIGateway.get_physical_objects_scenario(scenario_id, physical_object_function_id=4)
    return gdf[['geometry']].reset_index(drop=True)


async def _get_scenario_blocks(user_scenario_id: int, base_scenario_id: int,
                         boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    crs = boundaries.crs
    boundaries.geometry = boundaries.buffer(-1)

    water = await _get_scenario_water(user_scenario_id)
    water = water.to_crs(crs)
    user_roads = await _get_scenario_roads(user_scenario_id)
    user_roads = user_roads.to_crs(crs)
    base_roads =await _get_scenario_roads(base_scenario_id)
    base_roads = base_roads.to_crs(crs)
    roads = pd.concat([user_roads, base_roads]).reset_index(drop=True)
    roads.geometry = close_gaps(roads, 1)

    lines, polygons = preprocess_urban_objects(roads, None, water)
    blocks = cut_urban_blocks(boundaries, lines, polygons)
    return blocks


async def _get_scenario_info(scenario_id: int) -> tuple[int, int]:
    scenario = await UrbanAPIGateway.get_scenario(scenario_id)
    project_id = scenario['project']['project_id']
    project = await UrbanAPIGateway.get_project(project_id)
    base_scenario_id = project['base_scenario']['id']
    return project_id, base_scenario_id


async def _get_best_functional_zones_source(sources_df: pd.DataFrame) -> tuple[int | None, str | None]:
    sources = sources_df['source'].unique()
    for source in SOURCES_PRIORITY:
        if source in sources:
            sources_df = sources_df[sources_df['source'] == source]
            year = sources_df.year.max()
            return int(year), source
    return None, None  # FIXME ???


async def get_scenario_blocks(user_scenario_id: int):
    project_id, base_scenario_id = await _get_scenario_info(user_scenario_id)
    project_boundaries = await _get_project_boundaries(project_id)

    crs = project_boundaries.estimate_utm_crs()
    project_boundaries = project_boundaries.to_crs(crs)

    return await _get_scenario_blocks(user_scenario_id, base_scenario_id, project_boundaries)


async def get_scenario_functional_zones(scenario_id: int, source: str = None, year: int = None) -> gpd.GeoDataFrame:
    sources_df = await UrbanAPIGateway.get_functional_zones_sources_scenario(scenario_id)
    year, source = await _get_best_functional_zones_source(sources_df)
    functional_zones = await UrbanAPIGateway.get_functional_zones_scenario(scenario_id, year, source)
    return adapt_functional_zones(functional_zones)


async def get_scenario_buildings(scenario_id: int):
    try:
        gdf = await UrbanAPIGateway.get_physical_objects_scenario(scenario_id, physical_object_type_id=4, centers_only=True)
        gdf = adapt_buildings(gdf.reset_index(drop=True))
        crs = gdf.estimate_utm_crs()
        return impute_buildings(gdf.to_crs(crs)).to_crs(4326)
    except Exception as e:
        http_exception(404, f'No buildings found for scenario {scenario_id}', str(e))


async def get_scenario_services(scenario_id: int, service_types: pd.DataFrame):
    try:
        gdf = await UrbanAPIGateway.get_services_scenario(scenario_id, centers_only=True)
    except:
        return {}
    gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
    return {st: impute_services(gdf, st) for st, gdf in gdfs.items()}

