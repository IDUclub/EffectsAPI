import asyncio

import geopandas as gpd
import pandas as pd
from blocksnet.blocks.cutting import cut_urban_blocks, preprocess_urban_objects
from blocksnet.preprocessing.imputing import impute_buildings, impute_services

from app.clients.urban_api_client import UrbanAPIClient
from app.common.exceptions.http_exception_wrapper import http_exception
from app.common.utils.geodata import get_best_functional_zones_source
from app.effects_api.constants.const import LIVING_BUILDINGS_ID, ROADS_ID, WATER_ID
from app.effects_api.modules.buildings_service import adapt_buildings
from app.effects_api.modules.functional_sources_service import adapt_functional_zones
from app.effects_api.modules.scenario_service import close_gaps
from app.effects_api.modules.services_service import adapt_services

_SOURCES_PRIORITY = ["PZZ", "OSM", "User"]


async def _get_project_boundaries(
    project_id: int, token: str, client: UrbanAPIClient
) -> gpd.GeoDataFrame:
    geom = await client.get_project_geometry(project_id, token=token)
    return gpd.GeoDataFrame(geometry=[geom], crs=4326)


async def _get_context_boundaries(
    project_id: int, token: str, client: UrbanAPIClient
) -> gpd.GeoDataFrame:
    project = await client.get_project(project_id, token)
    context_ids = project["properties"]["context"]
    geometries = [
        await client.get_territory_geometry(territory_id)
        for territory_id in context_ids
    ]
    return gpd.GeoDataFrame(geometry=geometries, crs=4326)


async def _get_context_roads(scenario_id: int, token: str, client: UrbanAPIClient):
    gdf = await client.get_physical_objects(
        scenario_id, token, physical_object_function_id=ROADS_ID
    )
    return gdf[["geometry"]].reset_index(drop=True)


async def _get_context_water(scenario_id: int, token: str, client: UrbanAPIClient):
    gdf = await client.get_physical_objects(
        scenario_id, token=token, physical_object_function_id=WATER_ID
    )
    return gdf[["geometry"]].reset_index(drop=True)


async def _get_context_blocks(
    scenario_id: int, boundaries: gpd.GeoDataFrame, token: str, client: UrbanAPIClient
) -> gpd.GeoDataFrame:
    crs = boundaries.crs
    boundaries.geometry = boundaries.buffer(-1)

    water, roads = await asyncio.gather(
        _get_context_water(scenario_id, token, client),
        _get_context_roads(scenario_id, token, client),
    )

    water = water.to_crs(crs)
    roads = roads.to_crs(crs)
    roads.geometry = close_gaps(roads, 1)

    lines, polygons = preprocess_urban_objects(roads, None, water)
    blocks = cut_urban_blocks(boundaries, lines, polygons)
    return blocks


async def get_context_blocks(
    project_id: int, scenario_id: int, token: str, client: UrbanAPIClient
) -> gpd.GeoDataFrame:
    project_boundaries, context_boundaries = await asyncio.gather(
        _get_project_boundaries(project_id, token, client),
        _get_context_boundaries(project_id, token, client),
    )

    crs = context_boundaries.estimate_utm_crs()
    context_boundaries = context_boundaries.to_crs(crs)
    project_boundaries = project_boundaries.to_crs(crs)

    context_boundaries = context_boundaries.overlay(
        project_boundaries, how="difference"
    )
    return await _get_context_blocks(scenario_id, context_boundaries, token, client)


async def get_context_functional_zones(
    scenario_id: int,
    source: str | None,
    year: int | None,
    token: str,
    client: UrbanAPIClient,
) -> gpd.GeoDataFrame:
    sources_df = await client.get_functional_zones_sources(scenario_id, token)
    year, source = await get_best_functional_zones_source(sources_df, source, year)
    functional_zones = await client.get_functional_zones(
        scenario_id, year, source, token
    )
    functional_zones = functional_zones.loc[
        functional_zones.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
    ].reset_index(drop=True)
    return adapt_functional_zones(functional_zones)


async def get_context_buildings(scenario_id: int, token: str, client: UrbanAPIClient):
    gdf = await client.get_physical_objects(
        scenario_id,
        token,
        physical_object_type_id=LIVING_BUILDINGS_ID,
        centers_only=True,
    )
    if gdf is None or gdf.empty:
        raise http_exception(404, "No living buildings found for given scenario")
    gdf = adapt_buildings(gdf.reset_index(drop=True))
    crs = gdf.estimate_utm_crs()
    return impute_buildings(gdf.to_crs(crs)).to_crs(4326)


async def get_context_services(
    scenario_id: int, service_types: pd.DataFrame, token: str, client: UrbanAPIClient
):
    gdf = await client.get_services(scenario_id, token, centers_only=True)
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
    return {st: impute_services(gdf, st) for st, gdf in gdfs.items()}
