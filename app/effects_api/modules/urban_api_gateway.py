import json

import pandas as pd
import requests
import shapely
import geopandas as gpd

from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import urban_api_handler
from app.effects_api.constants import const
from app.effects_api.models import effects_models as em

#TODO context
# def get_physical_objects(project_id : int, **kwargs) -> dict:
#     res = api.get(f'/api/v1/projects/{project_id}/context/physical_objects_with_geometry', params=kwargs)
#     features = res['features']
#     return gpd.GeoDataFrame.from_features(features, crs=4326).set_index('physical_object_id')
#
# def get_services(project_id : int, **kwargs) -> dict:
#     res = api.get(f'/api/v1/projects/{project_id}/context/services_with_geometry', params=kwargs)
#     features = res['features']
#     return gpd.GeoDataFrame.from_features(features, crs=4326).set_index('service_id')
#
# def get_functional_zones_sources(project_id : int):
#     res = api.get(f'/api/v1/projects/{project_id}/context/functional_zone_sources')
#     return pd.DataFrame(res)
#
# def get_functional_zones(project_id : int, year : int, source : int):
#     res = api.get(f'/api/v1/projects/{project_id}/context/functional_zones', params={
#         'year': year,
#         'source': source
#     })
#     features = res['features']
#     return gpd.GeoDataFrame.from_features(features, crs=4326).set_index('functional_zone_id')
#
# def get_project(project_id : int) -> dict:
#     res = api.get(f'/api/v1/projects/{project_id}')
#     return res
#
# def get_project_geometry(project_id : int):
#     res = api.get(f'/api/v1/projects/{project_id}/territory')
#     geometry_json = json.dumps(res['geometry'])
#     return shapely.from_geojson(geometry_json)

#TODO scenario
def get_scenario(scenario_id : int) -> dict:
    res = urban_api_handler.get(f'/api/v1/scenarios/{scenario_id}')
    return res

def get_functional_zones_sources_scenario(scenario_id : int):
    res = urban_api_handler.get(f'/api/v1/scenarios/{scenario_id}/functional_zone_sources')
    return pd.DataFrame(res)

def get_functional_zones_scenario(scenario_id : int, year : int, source : int):
    res = urban_api_handler.get(f'/api/v1/scenarios/{scenario_id}/functional_zones', params={
        'year': year,
        'source': source
    })
    features = res['features']
    return gpd.GeoDataFrame.from_features(features, crs=4326).set_index('functional_zone_id')

def get_physical_objects_scenario(scenario_id : int, **kwargs):
    res = urban_api_handler.get(f'/api/v1/scenarios/{scenario_id}/physical_objects_with_geometry', params=kwargs)
    features = res['features']
    return gpd.GeoDataFrame.from_features(features, crs=4326).set_index('physical_object_id')

def get_services_scenario(scenario_id : int, **kwargs) -> dict:
    res = urban_api_handler.get(f'/api/v1/scenarios/{scenario_id}/services_with_geometry', params=kwargs)
    features = res['features']
    return gpd.GeoDataFrame.from_features(features, crs=4326).set_index('service_id')

