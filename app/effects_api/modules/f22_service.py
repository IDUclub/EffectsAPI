"""Здесь метод который собирает единый слой кварталов для сценраия и метод который считает параметры застройки
 https://github.com/vasilstar97/prostor-examples/blob/main/examples/Ф22.ipynb"""

import pandas as pd
import geopandas as gpd
from loguru import logger
from blocksnet.analysis.indicators import calculate_development_indicators
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.blocks.assignment import assign_land_use
from blocksnet.enums import LandUse
from blocksnet.machine_learning.regression import DensityRegressor
from blocksnet.relations import generate_adjacency_graph

from app.effects_api.modules.functional_sources_service import LAND_USE_RULES
from app.effects_api.modules.scenario_service import get_scenario_blocks, get_scenario_functional_zones, \
    get_scenario_buildings, get_scenario_services
from app.effects_api.modules.service_type_service import get_service_types


#собираем единый слой кварталов для сецнария
async def aggregate_blocks_layer_scenario(
        scenario_id: int,
        token: str = None,
        functional_zone_source: str = None,
        functional_zone_year: int = None) -> gpd.GeoDataFrame:
    logger.info(f"Aggregating blocks layer scenario {scenario_id}")
    blocks = await get_scenario_blocks(scenario_id)
    blocks_crs = blocks.crs
    logger.info(f"{len(blocks)} START blocks layer scenario{scenario_id}, CRS: {blocks.crs}")
    logger.info(f"Aggregating functional_zones layer scenario {scenario_id}")
    functional_zones = await get_scenario_functional_zones(scenario_id)
    functional_zones = functional_zones.to_crs(blocks_crs)
    logger.info(f"assign_land_use layer scenario {scenario_id}")
    blocks_lu = assign_land_use(blocks, functional_zones, LAND_USE_RULES)
    blocks = blocks.join(blocks_lu.drop(columns=['geometry']))
    logger.info(f"{len(blocks)} BLOCKS WITH LANDUSE blocks layer scenario {scenario_id}, CRS: {blocks.crs}, {blocks.columns}")
    logger.info(f"buildings layer scenario {scenario_id}")
    # #TODO жилых зданий может нге быть в сценарии, пока ломается здесь
    # # buildings = await get_scenario_buildings(scenario_id)
    logger.info(
        f"{len(blocks)} BLOCKS WITH BUILDINGS layer scenario {scenario_id}, CRS: {blocks.crs}, {blocks.columns}")
    buildings = buildings.to_crs(blocks_crs)
    if buildings is not None:
        buildings = buildings.to_crs(blocks.crs)
        blocks_buildings, _ = aggregate_objects(blocks, buildings)
        blocks = blocks.join(
            blocks_buildings.drop(columns=['geometry']).rename(columns={'objects_count': 'count_buildings'}))
        blocks['count_buildings'] = blocks['count_buildings'].fillna(0).astype(int)
    logger.info(f"service_types layer scenario {scenario_id}")
    service_types = await get_service_types()
    logger.info(
        f"{len(blocks)} BLOCKS WITH LANDUSE blocks layer scenario {scenario_id}, CRS: {blocks.crs}, {blocks.columns}")
    services_dict = await get_scenario_services(scenario_id, service_types)
    logger.info(
        f"{len(blocks)} BLOCKS WITH LANDUSE blocks layer scenario {scenario_id}, service_dict {services_dict}")
    for service_type, services in services_dict.items():
        services = services.to_crs(blocks.crs)
        blocks_services, _ = aggregate_objects(blocks, services)
        blocks_services['capacity'] = blocks_services['capacity'].fillna(0).astype(int)
        blocks_services['objects_count'] = blocks_services['objects_count'].fillna(0).astype(int)
        blocks = blocks.join(blocks_services.drop(columns=['geometry']).rename(columns={
            'capacity': f'capacity_{service_type}',
            'objects_count': f'count_{service_type}',
        }))
    logger.info(f"{len(blocks)} SERVICES blocks layer scenario {scenario_id}, CRS: {blocks.crs}")
    return blocks

async def get_services_layer(scenario_id: int):
    blocks = await get_scenario_blocks(scenario_id)
    blocks_crs = blocks.crs
    logger.info(f"{len(blocks)} START blocks layer scenario{scenario_id}, CRS: {blocks.crs}")
    service_types = await get_service_types()
    logger.info(f"{service_types}")
    services_dict = await get_scenario_services(scenario_id, service_types)

    for service_type, services in services_dict.items():
        services = services.to_crs(blocks_crs)
        blocks_services, _ = aggregate_objects(blocks, services)
        blocks_services['capacity'] = blocks_services['capacity'].fillna(0).astype(int)
        blocks_services['objects_count'] = blocks_services['objects_count'].fillna(0).astype(int)
        blocks = blocks.join(blocks_services.drop(columns=['geometry']).rename(columns={
            'capacity': f'capacity_{service_type}',
            'objects_count': f'count_{service_type}',
        }))
    logger.info(f"{len(blocks)} SERVICES blocks layer scenario {scenario_id}, CRS: {blocks.crs}")
    return blocks

async def run_development_parameters(scenario_id: int) -> gpd.GeoDataFrame | pd.DataFrame:
    blocks = await aggregate_blocks_layer_scenario(scenario_id)
    for lu in LandUse:
        blocks[lu.value] = blocks[lu.value].apply(lambda v: min(v, 1))
    logger.info(f"adjacency_graph scenario {scenario_id}")
    adjacency_graph = generate_adjacency_graph(blocks, 10)
    dr = DensityRegressor()

    logger.info(f"DensityRegressor scenario {scenario_id}")
    density_df = dr.evaluate(blocks, adjacency_graph)
    density_df.loc[density_df['fsi'] < 0, 'fsi'] = 0

    density_df.loc[density_df['gsi'] < 0, 'gsi'] = 0
    density_df.loc[density_df['gsi'] > 1, 'gsi'] = 1

    density_df.loc[density_df['mxi'] < 0, 'mxi'] = 0
    density_df.loc[density_df['mxi'] > 1, 'mxi'] = 1

    density_df.loc[blocks['residential'] == 0, 'mxi'] = 0
    density_df['site_area'] = blocks['site_area']

    logger.info(f"Calculating density indicators for {scenario_id}")
    development_df = calculate_development_indicators(density_df)
    development_df['population'] = development_df['living_area'] // 20

    mask = blocks['is_project']
    columns = ['build_floor_area', 'footprint_area', 'living_area', 'non_living_area', 'population']
    blocks.loc[mask, columns] = development_df.loc[mask, columns]
    return blocks
