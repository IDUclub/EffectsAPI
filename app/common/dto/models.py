from typing import Literal

from pydantic import BaseModel


class SourceYear(BaseModel):
    source: Literal["PZZ", "OSM", "User"]
    year: int
