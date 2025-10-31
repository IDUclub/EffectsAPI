from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.common.dto.models import FeatureCollectionModel


class ValuesTransformationSchema(BaseModel):
    geojson: FeatureCollectionModel = Field(
        ..., description="GeoJSON FeatureCollection for the scenario"
    )
