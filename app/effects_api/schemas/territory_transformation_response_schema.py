from typing import Optional, Dict

from pydantic import BaseModel, Field
from pydantic_geojson import FeatureCollectionModel


class TerritoryTransformationLayerResponse(BaseModel):
    """
    API response for a single service's territory transformation layer.
    Either 'before', 'after', or both can be present.
    Provision totals are optional numeric aggregates.
    """
    before: Optional[FeatureCollectionModel] = Field(
        None, description="GeoJSON FeatureCollection for the base (before) scenario"
    )
    after: Optional[FeatureCollectionModel] = Field(
        None, description="GeoJSON FeatureCollection for the transformed (after) scenario"
    )
    provision_total_before: Dict[str, float] = Field(
        None, description="Provision values for the base scenario, by service name"
    )
    provision_total_after: Optional[Dict[str, float]] = Field(
        None, description="Provision values for the transformed scenario, by service name"
    )