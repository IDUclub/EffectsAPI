from .dto.development_dto import DevelopmentDTO, ContextDevelopmentDTO


import pandas as pd
import geopandas as gpd
from blocksnet.machine_learning.regression import SocialRegressor
from loguru import logger
from blocksnet.analysis.indicators import calculate_development_indicators
from blocksnet.blocks.aggregation import aggregate_objects
from blocksnet.blocks.assignment import assign_land_use
from blocksnet.enums import LandUse
from blocksnet.machine_learning.regression import DensityRegressor
from blocksnet.relations import generate_adjacency_graph

from app.effects_api.modules.functional_sources_service import LAND_USE_RULES
from app.effects_api.modules.scenario_service import get_scenario_blocks, get_scenario_functional_zones, \
    get_scenario_services, get_scenario_buildings
from app.effects_api.modules.service_type_service import get_service_types
from .modules.context_service import get_context_blocks, get_context_functional_zones, get_context_buildings, \
    get_context_services
from .modules.task_api_service import get_project_id


class EffectsService:

    def __init__(self):
        pass

def _evaluate_master_plan(blocks: gpd.GeoDataFrame, buildings_blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    logger.info('Evaluating master plan effects')
    blocks = buildings_blocks_gdf.join(
        blocks.drop(columns="geometry"),
        how="left"
    )
    adjacency_graph = generate_adjacency_graph(blocks, 10)
    dr = DensityRegressor()
    density_df = dr.evaluate(blocks, adjacency_graph)


    density_df.loc[density_df['fsi'] < 0, 'fsi'] = 0
    density_df.loc[density_df['gsi'] < 0, 'gsi'] = 0
    density_df.loc[density_df['gsi'] > 1, 'gsi'] = 1
    density_df.loc[density_df['mxi'] < 0, 'mxi'] = 0
    density_df.loc[density_df['mxi'] > 1, 'mxi'] = 1
    density_df.loc[blocks['residential'] == 0, 'mxi'] = 0

    blocks['site_area'] = blocks.area
    density_df['site_area'] = blocks['site_area']
    development_df = calculate_development_indicators(density_df)
    development_df['population'] = development_df['living_area'] // 20
    cols = ['build_floor_area', 'footprint_area', 'living_area', 'non_living_area', 'population']
    blocks[cols] = development_df[cols].values
    for lu in LandUse:
        blocks[lu.value] = blocks[lu.value] * blocks['site_area']
    data = [blocks.drop(columns=['land_use', 'geometry']).sum().to_dict()]
    input = pd.DataFrame(data)

    input['latitude'] = blocks.geometry.union_all().centroid.x
    input['longitude'] = blocks.geometry.union_all().centroid.y
    input['buildings_count'] = input['count']
    sr = SocialRegressor()
    y_pred, pi_lower, pi_upper = sr.evaluate(input)
    iloc = 0
    result_data = {
        'pred': y_pred.apply(round).astype(int).iloc[iloc].to_dict(),
        'lower': pi_lower.iloc[iloc].to_dict(),
        'upper': pi_upper.iloc[iloc].to_dict(),
    }
    result_df = pd.DataFrame.from_dict(result_data)
    result_df['is_interval'] = (result_df['pred'] <= result_df['upper']) & (result_df['pred'] >= result_df['lower'])

    return result_df

async def aggregate_blocks_layer_scenario_context(
        scenario_id: int,
        functional_zone_source: str,
        functional_zone_year: int,
        context_functional_zone_source: str,
        context_functional_zone_year: int,
        token: str = None,
        ) -> gpd.GeoDataFrame:
    logger.info(f"Aggregating blocks layer scenario {scenario_id}")
    project_id = await get_project_id(scenario_id)

    scenario_blocks = await get_scenario_blocks(scenario_id)
    scenario_blocks_crs = scenario_blocks.crs
    scenario_blocks['site_area'] = scenario_blocks.area

    scenario_functional_zones = await get_scenario_functional_zones(scenario_id)
    scenario_functional_zones = scenario_functional_zones.to_crs(scenario_blocks_crs)
    scenario_blocks_lu = assign_land_use(scenario_blocks, scenario_functional_zones, LAND_USE_RULES)
    scenario_blocks = scenario_blocks.join(scenario_blocks_lu.drop(columns=['geometry']))

    scenario_buildings = await get_scenario_buildings(scenario_id)
    scenario_buildings = scenario_buildings.to_crs(scenario_blocks_crs)
    if scenario_buildings is not None:
        scenario_buildings = scenario_buildings.to_crs(scenario_blocks.crs)
        blocks_buildings, _ = aggregate_objects(scenario_blocks, scenario_buildings)
        scenario_blocks = scenario_blocks.join(
            blocks_buildings.drop(columns=['geometry']).rename(columns={'count': 'count_buildings'}))
        scenario_blocks['count_buildings'] = scenario_blocks['count_buildings'].fillna(0).astype(int)

    service_types = await get_service_types()
    scenario_services_dict = await get_scenario_services(scenario_id, service_types)

    for service_type, services in scenario_services_dict.items():
        services = services.to_crs(scenario_blocks.crs)
        scenario_blocks_services, _ = aggregate_objects(scenario_blocks, services)
        scenario_blocks_services['capacity'] = scenario_blocks_services['capacity'].fillna(0).astype(int)
        scenario_blocks_services['count'] = scenario_blocks_services['count'].fillna(0).astype(int)
        scenario_blocks = scenario_blocks.join(scenario_blocks_services.drop(columns=['geometry']).rename(columns={
            'capacity': f'capacity_{service_type}',
            'count': f'count_{service_type}',
        }))

    #---------------------------------------------------------#

    context_blocks = await get_context_blocks(project_id)
    context_blocks = context_blocks.to_crs(scenario_blocks_crs)
    context_blocks['site_area'] = context_blocks.area

    context_functional_zones = await get_context_functional_zones(project_id)
    context_functional_zones = context_functional_zones.to_crs(scenario_blocks_crs)
    context_blocks_lu = assign_land_use(context_blocks, context_functional_zones, LAND_USE_RULES)
    context_blocks = context_blocks.join(context_blocks_lu.drop(columns=['geometry']))

    context_buildings = await get_context_buildings(project_id)
    context_buildings = context_buildings.to_crs(scenario_blocks_crs)
    if context_buildings is not None:
        context_buildings = context_buildings.to_crs(context_blocks.crs)
        context_blocks_buildings, _ = aggregate_objects(context_blocks, context_buildings)
        context_blocks = context_blocks.join(
            context_blocks_buildings.drop(columns=['geometry']).rename(columns={'count': 'count_buildings'}))
        context_blocks['count_buildings'] = context_blocks['count_buildings'].fillna(0).astype(int)

    context_services_dict = await get_context_services(project_id, service_types)
    for service_type, services in context_services_dict.items():
        services = services.to_crs(context_blocks.crs)
        context_blocks_services, _ = aggregate_objects(scenario_blocks, services)
        context_blocks_services['capacity'] = context_blocks_services['capacity'].fillna(0).astype(int)
        context_blocks_services['count'] = context_blocks_services['count'].fillna(0).astype(int)
        context_blocks = context_blocks.join(context_blocks_services.drop(columns=['geometry']).rename(columns={
            'capacity': f'capacity_{service_type}',
            'count': f'count_{service_type}',
        }))
    buildings_gdf  = scenario_buildings.join(context_buildings.drop(columns=['geometry']).rename(columns={}))
    blocks = context_blocks.join(scenario_blocks)

    return blocks

async def aggregate_blocks_layer_scenario(
        scenario_id: int,
        token: str = None,
        functional_zone_source: str = None,
        functional_zone_year: int = None) -> gpd.GeoDataFrame:
    logger.info(f"Aggregating blocks layer scenario {scenario_id}")
    blocks = await get_scenario_blocks(scenario_id)
    blocks_crs = blocks.crs
    blocks['site_area'] = blocks.area
    functional_zones = await get_scenario_functional_zones(scenario_id)
    functional_zones = functional_zones.to_crs(blocks_crs)
    blocks_lu = assign_land_use(blocks, functional_zones, LAND_USE_RULES)
    blocks = blocks.join(blocks_lu.drop(columns=['geometry']))
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


    # mask = blocks['is_project']
    # columns = ['build_floor_area', 'footprint_area', 'living_area', 'non_living_area', 'population']
    # blocks.loc[mask, columns] = development_df.loc[mask, columns]
    return development_df

    async def calc_project_development(self, params: DevelopmentDTO):
        """
        Function calculates development only for project with blocksnet
        Args:
            params (DevelopmentDTO):
        Returns:
            --
        """

        pass

    async def calc_context_development(self, params: ContextDevelopmentDTO):
        """
        Function calculates development for context  with project with blocksnet
        Args:
            params (DevelopmentDTO):
        Returns:
            --
        """

        pass
