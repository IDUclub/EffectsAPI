from typing import Dict, Literal

from pydantic import BaseModel, Field

from app.common.dto.models import SourceYear


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


class SocioEconomicByProjectComputedDTO(SocioEconomicByProjectDTO):
    context_func_zone_source: Literal["PZZ", "OSM", "User"]
    context_func_source_year: int
    project_sources: Dict[int, SourceYear]
