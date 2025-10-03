from typing import Literal, Optional

from pydantic import BaseModel, Field


class DevelopmentDTO(BaseModel):
    force: bool = Field(
        default=False, description="flag for recalculating the scenario"
    )

    scenario_id: int = Field(
        ...,
        examples=[822],
        description="Project-scenario ID to retrieve data from.",
    )
    proj_func_zone_source: Optional[Literal["PZZ", "OSM", "User"]] = Field(
        None,
        examples=["User", "PZZ", "OSM"],
        description=(
            "Preferred source for project functional zones. "
            "Default priority: User → PZZ → OSM."
        ),
    )
    proj_func_source_year: Optional[int] = Field(
        None,
        examples=[2023, 2024],
        description="Year of the chosen project functional-zone source.",
    )


class ContextDevelopmentDTO(DevelopmentDTO):
    context_func_zone_source: Optional[Literal["PZZ", "OSM", "User"]] = Field(
        None,
        examples=["PZZ", "OSM"],
        description=(
            "Preferred source for context functional zones. "
            "Default priority: PZZ → OSM."
        ),
    )
    context_func_source_year: Optional[int] = Field(
        None,
        examples=[2023, 2024],
        description="Year of the chosen context functional-zone source.",
    )
