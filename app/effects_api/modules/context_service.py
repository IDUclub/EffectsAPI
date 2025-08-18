import geopandas as gpd
import pandas as pd
from blocksnet.blocks.cutting import cut_urban_blocks, preprocess_urban_objects
from blocksnet.preprocessing.imputing import impute_buildings, impute_services

from app.dependencies import urban_api_gateway
from app.effects_api.constants.const import LIVING_BUILDINGS_ID, ROADS_ID, WATER_ID
from app.effects_api.modules.buildings_service import adapt_buildings
from app.effects_api.modules.functional_sources_service import adapt_functional_zones
from app.effects_api.modules.scenario_service import (
    _get_best_functional_zones_source,
    close_gaps,
)
from app.effects_api.modules.services_service import adapt_services

# from blocksnet.relations import get_accessibility_context
# from blocksnet.config import service_types_config
# from blocksnet.analysis.provision import competitive_provision


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


async def _get_context_roads(scenario_id: int):
    gdf = await urban_api_gateway.get_physical_objects(
        scenario_id, physical_object_function_id=ROADS_ID
    )
    return gdf[["geometry"]].reset_index(drop=True)


async def _get_context_water(scenario_id: int):
    gdf = await urban_api_gateway.get_physical_objects(
        scenario_id, physical_object_function_id=WATER_ID
    )
    return gdf[["geometry"]].reset_index(drop=True)


async def _get_context_blocks(
    scenario_id: int, boundaries: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    crs = boundaries.crs
    boundaries.geometry = boundaries.buffer(-1)

    water = await _get_context_water(scenario_id)
    water = water.to_crs(crs)
    roads = await _get_context_roads(scenario_id)
    roads = roads.to_crs(crs)
    roads.geometry = close_gaps(roads, 1)

    lines, polygons = preprocess_urban_objects(roads, None, water)
    blocks = cut_urban_blocks(boundaries, lines, polygons)
    return blocks


async def get_context_blocks(project_id: int, scenario_id: int):
    project_boundaries = await _get_project_boundaries(project_id)
    context_boundaries = await _get_context_boundaries(project_id)

    crs = context_boundaries.estimate_utm_crs()
    context_boundaries = context_boundaries.to_crs(crs)
    project_boundaries = project_boundaries.to_crs(crs)

    context_boundaries = context_boundaries.overlay(
        project_boundaries, how="difference"
    )
    return await _get_context_blocks(scenario_id, context_boundaries)


async def get_context_functional_zones(
    scenario_id: int, source: str, year: int, token: str
) -> gpd.GeoDataFrame:
    sources_df = await urban_api_gateway.get_functional_zones_sources(scenario_id)
    year, source = await _get_best_functional_zones_source(sources_df, source, year)
    functional_zones = await urban_api_gateway.get_functional_zones(
        scenario_id, year, source
    )
    functional_zones = functional_zones.loc[
        functional_zones.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
    ].reset_index(drop=True)
    return adapt_functional_zones(functional_zones)


async def get_context_buildings(scenario_id: int):
    gdf = await urban_api_gateway.get_physical_objects(
        scenario_id, physical_object_type_id=LIVING_BUILDINGS_ID, centers_only=True
    )
    gdf = adapt_buildings(gdf.reset_index(drop=True))
    crs = gdf.estimate_utm_crs()
    return impute_buildings(gdf.to_crs(crs)).to_crs(4326)


async def get_context_services(scenario_id: int, service_types: pd.DataFrame):
    gdf = await urban_api_gateway.get_services(scenario_id, centers_only=True)
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdfs = adapt_services(gdf.reset_index(drop=True), service_types)
    return {st: impute_services(gdf, st) for st, gdf in gdfs.items()}


# async def _assess_provision(
#         blocks: pd.DataFrame, acc_mx: pd.DataFrame, service_type: str
# ) -> gpd.GeoDataFrame:
#     _, demand, accessibility = service_types_config[service_type].values()
#     blocks["is_project"] = (
#         blocks["is_project"]
#         .fillna(False)
#         .astype(bool)
#     )
#     context_ids = await _get_accessibility_context(blocks, acc_mx, accessibility)
#     capacity_column = f"capacity_{service_type}"
#     if capacity_column not in blocks.columns:
#         blocks_df = blocks[["geometry", "population"]].fillna(0)
#         blocks_df["capacity"] = 0
#     else:
#         blocks_df = blocks.rename(columns={capacity_column: "capacity"})[
#             ["geometry", "population", "capacity"]
#         ].fillna(0)
#     prov_df, _ = competitive_provision(blocks_df, acc_mx, accessibility, demand)
#     prov_df = prov_df.loc[context_ids].copy()
#     return blocks[["geometry"]].join(prov_df, how="right")
#
# async def _get_accessibility_context(
#     blocks: pd.DataFrame, acc_mx: pd.DataFrame, accessibility: float
# ) -> list[int]:
#     blocks["population"] = blocks["population"].fillna(0)
#     project_blocks = blocks[blocks["is_project"]].copy()
#     context_blocks = get_accessibility_context(
#         acc_mx, project_blocks, accessibility, out=False, keep=True
#     )
#     return list(context_blocks.index)
