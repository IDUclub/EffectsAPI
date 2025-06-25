import json

import pandas as pd
import requests
import shapely
import geopandas as gpd

from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import urban_api_handler
from app.effects_api.constants import const
from app.effects_api.models import effects_models as em


def get_zones(scenario_id: int, token, is_context: bool = False) -> gpd.GeoDataFrame:
  """

  Args:
    scenario_id (int): scenario id

  Returns:
    gpd.GeoDataFrame: geodataframe with zones

  """

  def _form_source_params(sources: list[dict]) -> dict:
    source_names = [i["source"] for i in sources]
    source_data_df = pd.DataFrame(sources)
    if "PZZ" in source_names:
      return source_data_df.loc[
        source_data_df[source_data_df["source"] == "PZZ"]["year"].idxmax()
      ].to_dict()
    elif "OSM" in source_names:
      return source_data_df.loc[
        source_data_df[source_data_df["source"] == "OSM"]["year"].idxmax()
      ].to_dict()
    elif "User" in source_names:
      return source_data_df.loc[
        source_data_df[source_data_df["source"] == "User"]["year"].idxmax()
      ].to_dict()
    else:
      raise http_exception(404, "Source type not found for given ID", scenario_id)

  zones_sources = urban_api_handler.get(
    url=f"{const.URBAN_API}/api/v1/scenarios/{scenario_id}/functional_zone_sources",
  )
  zones_params_request = _form_source_params(zones_sources.json())

  if is_context:
    project_id = get_project_info(scenario_id, token)["project_id"]
    target_zones = urban_api_handler.get(
      url=f"{const.URBAN_API}/api/v1/projects/{project_id}/context/functional_zones",
      params=zones_params_request,)
  else:
    target_zones = urban_api_handler.get(
      url=f"{const.URBAN_API}/api/v1/scenarios/{scenario_id}/functional_zones",
      params=zones_params_request,
    )
  target_zones_gdf = gpd.GeoDataFrame.from_features(target_zones.json(), crs=4326)
  target_zones_gdf["functional_zone"] = target_zones_gdf["functional_zone_type"].apply(lambda x: x.get("name"))
  return target_zones_gdf

def get_scenarios_by_project_id(project_id: int, token: str) -> dict:
    res = urban_api_handler.get(const.URBAN_API + f'/api/v1/projects/{project_id}/scenarios',
                                headers={'Authorization': f'Bearer {token}'})
    res.raise_for_status()
    return res.json()

def get_project_geometry(project_id: int, token: str):
        """
        Fetches the territory information for a project.

        Parameters:
        project_id (int): ID of the project.

        Returns:
        dict: Territory information.
        """
        endpoint = f"/api/v1/projects/{project_id}/territory"

        response = urban_api_handler.get(endpoint, headers = {
            "Authorization": f"Bearer {token}"""
        })

        if not response:
            raise http_exception(404, f"No territory information found for project ID:", project_id)

        return response

def get_scenario(scenario_id : int, token: str) -> dict:
    res = urban_api_handler.get(f'/api/v1/scenarios/{scenario_id}',
                                headers={'Authorization': f'Bearer {token}'})
    return res


def get_based_scenario_id(project_info, token):
    scenarios = get_scenarios_by_project_id(project_info['project_id'], token)
    based_scenario_id = list(filter(lambda x: x['is_based'], scenarios))[0]['scenario_id']
    return based_scenario_id


def _get_scenario_objects(
        scenario_id: int,
        token: str,
        is_context: bool,
        project_id: int = None,
        physical_object_type_id: int | None = None,
        service_type_id: int | None = None,
        physical_object_function_id: int | None = None,
        urban_function_id: int | None = None,
):
    headers = {'Authorization': f'Bearer {token}'}
    if is_context:
        url = const.URBAN_API + f'/api/v1/projects/{project_id}/context/geometries_with_all_objects'
    else:
        url = const.URBAN_API + f'/api/v1/scenarios/{scenario_id}/geometries_with_all_objects'
    res = urban_api_handler.get(url, params={
        'physical_object_type_id': physical_object_type_id,
        'service_type_id': service_type_id,
        'physical_object_function_id': physical_object_function_id,
        'urban_function_id': urban_function_id
    }, headers=headers)
    return res.json()


def get_physical_objects(scenario_id: int, token: str, is_context: bool, *args, **kwargs) -> gpd.GeoDataFrame:
    if is_context:
        project_id = get_project_info(scenario_id, token)["project_id"]
        raw = _get_scenario_objects(
            scenario_id, token, is_context, project_id, *args, **kwargs
        )
    else:
        raw = _get_scenario_objects(
            scenario_id, token, is_context, *args, **kwargs
        )

    if isinstance(raw, dict):
        collections = [raw]
    else:
        collections = raw  # type: ignore

    features: list[dict[str, any]] = [
        feature
        for collection in collections
        for feature in collection.get("features", [])
    ]

    return gpd.GeoDataFrame.from_features(features, crs=4326).set_index('physical_object_id')


def get_services(project_scenario_id: int, token: str, is_context: bool) -> gpd.GeoDataFrame:
    if is_context:
        project_id = get_project_info(project_scenario_id, token)["project_id"]
        res = urban_api_handler.get(f'/api/v1/projects/{project_id}/context/services_with_geometry',
                                    headers={'Authorization': f'Bearer {token}'})
    else:
        res = urban_api_handler.get(f'/api/v1/scenarios/{project_scenario_id}/services_with_geometry',
                                    headers={'Authorization': f'Bearer {token}'})
    features = res['features']
    return gpd.GeoDataFrame.from_features(features, crs=4326).set_index('service_id')


def get_physical_object_types():
    res = urban_api_handler.get(const.URBAN_API + f'/api/v1/physical_object_types', verify=False)
    return res.json()


def _get_scenario_by_id(scenario_id: int, token: str) -> dict:
    res = urban_api_handler.get(const.URBAN_API + f'/api/v1/scenarios/{scenario_id}',
                                headers={'Authorization': f'Bearer {token}'})
    res.raise_for_status()
    return res.json()


def _get_project_territory_by_id(project_id: int, token: str) -> dict:
    res = urban_api_handler.get(const.URBAN_API + f'/api/v1/projects/{project_id}/territory',
                                headers={'Authorization': f'Bearer {token}'})
    res.raise_for_status()
    return res.json()


def _get_project_by_id(project_id: int, token: str) -> dict:
    res = urban_api_handler.get(const.URBAN_API + f'/api/v1/projects/{project_id}',
                                headers={'Authorization': f'Bearer {token}'})
    res.raise_for_status()
    return res.json()


def _get_territory_by_id(territory_id: int) -> dict:
    res = urban_api_handler.get(const.URBAN_API + f'/api/v1/territory/{territory_id}')
    res.raise_for_status()
    return res.json()


def _get_context_geometry(territories_ids: list[int]):
    geometries = []
    for territory_id in territories_ids:
        territory = _get_territory_by_id(territory_id)
        geom_json = json.dumps(territory['geometry'])
        geometry = shapely.from_geojson(geom_json)
        geometries.append(geometry)
    return shapely.unary_union(geometries)


def get_project_info(project_scenario_id: int, token: str) -> dict:
    """
    Fetch project data
    """
    scenario_info = _get_scenario_by_id(project_scenario_id, token)
    is_based = scenario_info['is_based']
    project_id = scenario_info['project']['project_id']

    project_info = _get_project_by_id(project_id, token)
    context_ids = project_info['properties']['context']

    project_territory = _get_project_territory_by_id(project_id, token)
    region_id = project_territory['project']['region']['id']
    project_geometry = json.dumps(project_territory['geometry'])

    return {
        'project_id': project_id,
        'region_id': region_id,
        'is_based': is_based,
        'geometry': shapely.from_geojson(project_geometry),
        'context': _get_context_geometry(context_ids)
    }

