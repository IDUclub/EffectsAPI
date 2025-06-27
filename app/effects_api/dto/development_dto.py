from typing import Literal

from pydantic import BaseModel, Field


class DevelopmentDTO(BaseModel):

    scenario_id: Field(..., examples=[822], description="Project scenario id to retrieve data from.")
    proj_func_zone_source: Literal["PZZ", "OSM", "User"] = Field(
        default=None,
        examples=["User", "PZZ", "OSM",],
        description="Project functional zones source to retrieve data from. Default retrieves in priority User -> PZZ -> OSM"
    )
    proj_func_source_year: int = Field(
        default=None,
        examples=[2023, 2024],
        description="Project functional zones source year to retrieve data from. As default retrieves latest year."
    )


class ContextDevelopmentDTO(DevelopmentDTO):

    context_func_zone_source: Literal["PZZ", "OSM", "User"] = Field(
        default=None,
        examples=["PZZ", "OSM"],
        description="Project functional zones source to retrieve data from. As default retrieves in priority PZZ -> OSM"
    )
    context_func_source_year: int = Field(
        default=None,
        examples=[2023, 2024],
        description="Context functional zones source year to retrieve data from. Default retrieves latest year."
    )
