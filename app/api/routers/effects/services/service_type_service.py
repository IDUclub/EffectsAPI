import pandas as pd
import geopandas as gpd
import requests
from fastapi import HTTPException

from app.api.utils import const
from blocksnet.models import ServiceType

def _get_service_types(region_id : int) -> pd.DataFrame:
  res = requests.get(const.URBAN_API + f'/api/v1/territory/{region_id}/service_types')
  res.raise_for_status()
  df = pd.DataFrame(res.json())
  return df.set_index('service_type_id')

def _get_normatives(region_id : int) -> pd.DataFrame:
  res = requests.get(const.URBAN_API + f'/api/v1/territory/{region_id}/normatives', params={'year': const.NORMATIVES_YEAR})
  res.raise_for_status()
  df = pd.DataFrame(res.json())
  df['service_type_id'] = df['service_type'].apply(lambda st : st['id'])
  return df.set_index('service_type_id')

def get_bn_service_types(region_id : int) -> list[ServiceType]:
  """
  Befriend normatives and service types into BlocksNet format
  """
  db_service_types_df = _get_service_types(region_id)
  db_normatives_df = _get_normatives(region_id)
  service_types_df = db_service_types_df.merge(db_normatives_df, left_index=True, right_index=True)
  # filter by minutes not null
  service_types_df = service_types_df[~service_types_df['time_availability_minutes'].isna()]
  # filter by capacity not null
  service_types_df = service_types_df[~service_types_df['services_capacity_per_1000_normative'].isna()]
  
  service_types = []
  for _, row in service_types_df.iterrows():
    service_type = ServiceType(
      code=row['code'], 
      name=row['name'], 
      accessibility=row['time_availability_minutes'],
      demand=row['services_capacity_per_1000_normative'],
      land_use = [], #TODO
      bricks = [] #TODO
    )
    service_types.append(service_type)
  return service_types

def get_zones(scenario_id: int) -> gpd.GeoDataFrame:
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
      raise HTTPException(status_code=404, detail={"msg": "Pzz zones not found. Upload pzz firstly"})

  zones_sources = requests.get(
    url=f"{const.URBAN_API}/api/v1/scenarios/{scenario_id}/functional_zone_sources",
  )
  zones_params_request = _form_source_params(zones_sources.json())
  target_zones = requests.get(
    url=f"{const.URBAN_API}/api/v1/scenarios/{scenario_id}/functional_zones",
    params=zones_params_request,
  )
  target_zones_gdf = gpd.GeoDataFrame.from_features(target_zones.json(), crs=4326)
  target_zones_gdf["zone"] = target_zones_gdf["functional_zone_type"].apply(lambda x: x.get("name"))
  return target_zones_gdf
