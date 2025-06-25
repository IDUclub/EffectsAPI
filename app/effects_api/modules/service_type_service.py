import pandas as pd
import geopandas as gpd
from fastapi import HTTPException

from app.dependencies import urban_api_handler
from app.effects_api.constants import const
from app.effects_api.modules.urban_api_gateway import get_project_info


async def get_indicators(parent_id: int | None = None, **kwargs):
  res = urban_api_handler.get('/api/v1/indicators_by_parent', params={
    'parent_id': parent_id,
    **kwargs
  })
  return pd.DataFrame(res).set_index('indicator_id')


async def get_service_types(**kwargs):
  res = urban_api_handler.get('/api/v1/service_types', params=kwargs)
  return pd.DataFrame(res).set_index('service_type_id')


async def get_social_values(**kwargs):
  res = urban_api_handler.get('/api/v1/social_values', params=kwargs)
  return pd.DataFrame(res).set_index('soc_value_id')


async def get_social_value_service_types(soc_value_id: int, **kwargs):
  res = urban_api_handler.get(f'/api/v1/social_values/{soc_value_id}/service_types', params=kwargs)
  if len(res) == 0:
    return None
  return pd.DataFrame(res).set_index('service_type_id')


async def get_service_type_social_values(service_type_id: int, **kwargs):
  res = urban_api_handler.get(f'/api/v1/service_types/{service_type_id}/social_values', params=kwargs)
  if len(res) == 0:
    return None
  return pd.DataFrame(res).set_index('soc_value_id')


