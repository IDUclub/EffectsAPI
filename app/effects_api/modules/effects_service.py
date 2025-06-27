import os

import geopandas as gpd
import warnings
import pandas as pd
from blocksnet.analysis.indicators import calculate_development_indicators
from blocksnet.enums import LandUse
from blocksnet.machine_learning.regression import DensityRegressor, SocialRegressor
from blocksnet.relations import generate_adjacency_graph
from urllib3.exceptions import InsecureRequestWarning
from loguru import logger
from app.effects_api.constants import const
from app.effects_api.models import effects_models as em
from app.effects_api.modules import blocksnet_service as bs
from app.gateways import urban_api_gateway as ps

for warning in [pd.errors.PerformanceWarning, RuntimeWarning, pd.errors.SettingWithCopyWarning, InsecureRequestWarning,
                FutureWarning]:
    warnings.filterwarnings(action='ignore', category=warning)

PROVISION_COLUMNS = ['provision', 'demand', 'demand_within']


def _get_file_path(project_scenario_id: int, effect_type: em.EffectType, scale_type: em.ScaleType):
    file_path = f'{project_scenario_id}_{effect_type.name}_{scale_type.name}'
    return os.path.join(const.DATA_PATH, f'{file_path}.parquet')

def _evaluate_master_plan(blocks: gpd.GeoDataFrame, buildings_blocks_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    logger.info('Evaluating master plan effects')
    blocks = buildings_blocks_gdf.join(
        blocks.drop(columns="geometry"),  # не тащим геометрию дважды
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


def _evaluation_exists(project_scenario_id: int, token: str):
    exists = True
    for effect_type in list(em.EffectType):
        for scale_type in list(em.ScaleType):
            file_path = _get_file_path(project_scenario_id, effect_type, scale_type)
            if not os.path.exists(file_path):
                exists = False
    return exists


def delete_evaluation(project_scenario_id: int):
    for effect_type in list(em.EffectType):
        for scale_type in list(em.ScaleType):
            file_path = _get_file_path(project_scenario_id, effect_type, scale_type)
            if os.path.exists(file_path):
                os.remove(file_path)


def evaluate_f22(project_scenario_id: int, token: str, reevaluate: bool = True, is_context: bool = False):
    logger.info(f'Fetching {project_scenario_id} project info')

    project_info = ps.get_project_info(project_scenario_id, token)
    based_scenario_id = ps.get_based_scenario_id(project_info, token)
    if project_scenario_id != based_scenario_id:
        evaluate_f22(based_scenario_id, token, reevaluate=False)

    exists = _evaluation_exists(project_scenario_id, token)
    if exists and not reevaluate:
        logger.info(f'{project_scenario_id} evaluation already exists')
        return

    logger.info('Fetching region service types')
    logger.info('Fetching physical object types')
    physical_object_types = ps.get_physical_object_types()

    logger.info('Fetching project model')
    blocks_gdf, buildings_blocks_gdf = bs.fetch_city_model(
        project_info=project_info,
        project_scenario_id=project_scenario_id,
        token=token,
        # service_types=service_types,
        physical_object_types=physical_object_types,
        is_context=is_context
    )


    _evaluate_master_plan(blocks_gdf, buildings_blocks_gdf)
