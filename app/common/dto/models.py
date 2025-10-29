from typing import Literal

from pydantic import BaseModel, Field


class SourceYear(BaseModel):
    source: Literal["PZZ", "OSM", "User"]
    year: int


class ServiceType(BaseModel):
    id: int
    name: str
