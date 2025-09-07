from typing import Literal, Optional, Dict

from pydantic import BaseModel, Field


class DevelopmentDTO(BaseModel):
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


class SocioEconomicByScenarioDTO(ContextDevelopmentDTO):

    split: bool = Field(
        default=False,
        examples=[False, True],
        description="If split will return additional evaluation for each context mo",
    )


class SocioEconomicByProjectDTO(BaseModel):
    project_id: int = Field(
        ...,
        examples=[120],
        description="Project ID to retrieve data from.",
    )

    regional_scenario_id: int = Field(
        ...,
        examples=[122],
        description="Regional scenario ID using for filtering.",
    )

    split: bool = Field(
        default=False,
        examples=[False, True],
        description="If split will return additional evaluation for each context mo",
    )

class SourceYear(BaseModel):
    source: Literal["PZZ", "OSM", "User"]
    year: int

class SocioEconomicByProjectComputedDTO(SocioEconomicByProjectDTO):
    context_func_zone_source: Literal["PZZ", "OSM", "User"]
    context_func_source_year: int
    project_sources: Dict[int, SourceYear]