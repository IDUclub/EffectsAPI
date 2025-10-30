from typing import Dict, Optional, List

from pydantic import BaseModel, Field

from app.common.dto.models import FeatureCollectionModel


class ValuesTransformationSchema(BaseModel):
    geojson: FeatureCollectionModel = Field(
        None, description="GeoJSON FeatureCollection for the scenario")