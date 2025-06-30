import geopandas as gpd
import pandas as pd
from blocksnet.blocks.cutting import cut_urban_blocks, preprocess_urban_objects
from blocksnet.preprocessing.imputing import impute_buildings, impute_services

from app.dependencies import urban_api_gateway
from app.effects_api.modules.buildings_service import adapt_buildings
from app.effects_api.modules.functional_sources_service import \
    adapt_functional_zones
from app.effects_api.modules.scenario_service import (
    _get_best_functional_zones_source, close_gaps)
from app.effects_api.modules.services_service import adapt_services


async def _get_project_boundaries(project_id: int):
    return gpd.GeoDataFrame(
        geometry=[await urban_api_gateway.get_project_geometry(project_id)], crs=4326
    )


async def _get_context_boundaries(project_id: int) -> gpd.GeoDataFrame:
    project = await urban_api_gateway.get_project(project_id)
    context_ids = project["properties"]["context"]
    geometries = [
        await urban_api_gateway.get_territory_geometry(territory_id)
        for territory_id in context_ids
    ]
    return gpd.GeoDataFrame(geometry=geometries, crs=4326)


async def _get_context_roads(project_id: int):
    gdf = await urban_api_gateway.get_physical_objects(
        project_id, physical_object_function_id=26
    )
    return gdf[["geometry"]].reset_index(drop=True)


async def _get_context_water(project_id: int):
    gdf = await urban_api_gateway.get_physical_objects(
        project_id, physical_object_function_id=4
    )
    return gdf[["geometry"]].reset_index(drop=True)


async def _get_context_blocks(
    project_id: int, boundaries: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    crs = boundaries.crs
    boundaries.geometry = boundaries.buffer(-1)

    water = await _get_context_water(project_id)
    water = water.to_crs(crs)
    roads = await _get_context_roads(project_id)
    roads = roads.to_crs(crs)
    roads.geometry = close_gaps(roads, 1)

    lines, polygons = preprocess_urban_objects(roads, None, water)
    blocks = cut_urban_blocks(boundaries, lines, polygons)
    return blocks


async def get_context_blocks(project_id: int):
    project_boundaries = await _get_project_boundaries(project_id)
    context_boundaries = await _get_context_boundaries(project_id)

    crs = context_boundaries.estimate_utm_crs()
    context_boundaries = context_boundaries.to_crs(crs)
    project_boundaries = project_boundaries.to_crs(crs)

    context_boundaries = context_boundaries.overlay(
        project_boundaries, how="difference"
    )
    return await _get_context_blocks(project_id, context_boundaries)


async def get_context_functional_zones(
    project_id: int, source: str, year: int, token: str
) -> gpd.GeoDataFrame:
    sources_df = await urban_api_gateway.get_functional_zones_sources(project_id)
    year, source = await _get_best_functional_zones_source(sources_df, source, year)
    # year, source = await urban_api_gateway.get_optimal_func_zone_request_data(token, scenario_id, year, source)
    functional_zones = await urban_api_gateway.get_functional_zones(
        project_id, year, source
    )
    return adapt_functional_zones(functional_zones)


async def get_context_buildings(project_id: int):
    gdf = await urban_api_gateway.get_physical_objects(
        project_id, physical_object_type_id=4, centers_only=True
    )
    gdf = adapt_buildings(gdf.reset_index(drop=True))
    crs = gdf.estimate_utm_crs()
    return impute_buildings(gdf.to_crs(crs)).to_crs(4326)


async def get_context_services(project_id: int, service_types: pd.DataFrame):
    gdf = await urban_api_gateway.get_services(project_id, centers_only=True)
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
    return {st: impute_services(gdf, st) for st, gdf in gdfs.items()}
