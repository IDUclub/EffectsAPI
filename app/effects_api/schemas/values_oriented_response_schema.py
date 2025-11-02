from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.common.dto.models import FeatureCollectionModel


class ValuesOrientedResponseSchema(BaseModel):
    base_scenario_id: int = Field(..., description="Id of the base scenario")
    geojson: FeatureCollectionModel = Field(
        ..., description="GeoJSON FeatureCollection for the base scenario"
    )
    values_table: Dict[str, Dict[str, str | float]] = Field(
        ..., description="Values table for the base scenario"
    )
    services_type_deficit: List[Dict[str, str | float | List[str]]] = Field(
        ..., description="Services type deficit for the base scenario"
    )
