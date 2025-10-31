from typing import List, Literal, Union

from pydantic import BaseModel
from pydantic_geojson import FeatureModel, MultiPolygonModel, PolygonModel
from pydantic_geojson._base import FeatureCollectionFieldType


class SourceYear(BaseModel):
    source: Literal["PZZ", "OSM", "User"]
    year: int


class ServiceType(BaseModel):
    id: int
    name: str


class FeatureCollectionModel(BaseModel):
    type: str = FeatureCollectionFieldType
    features: List[Union[PolygonModel, MultiPolygonModel, FeatureModel],]
