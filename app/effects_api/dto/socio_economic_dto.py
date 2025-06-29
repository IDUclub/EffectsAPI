from typing import Literal, Optional
from pydantic import BaseModel, Field


class SocioEconomicDTO(BaseModel):
    """Input payload for socio-economic evaluation endpoint."""
    scenario_id: int = Field(
        ...,
        examples=[788],
        description="Project-scenario ID to retrieve data from.",
    )
    project_functional_zone_source: Optional[Literal["PZZ", "OSM", "User"]] = Field(
        None,
        examples=["User", "PZZ", "OSM"],
        description=(
            "Preferred source for PROJECT functional zones. "
            "If omitted: priority User, PZZ, OSM."
        ),
    )
    project_functional_zone_year: Optional[int] = Field(
        None,
        examples=[2023, 2024],
        description=(
            "Year of the chosen PROJECT functional-zone source. "
            "If omitted, the newest year is used."
        ),
    )
    context_functional_zone_source: Optional[Literal["PZZ", "OSM", "User"]] = Field(
        None,
        examples=["PZZ", "OSM"],
        description=(
            "Preferred source for CONTEXT functional zones. "
            "If omitted: priority PZZ → OSM."
        ),
    )
    context_functional_zone_year: Optional[int] = Field(
        None,
        examples=[2023, 2024],
        description=(
            "Year of the chosen CONTEXT functional-zone source. "
            "If omitted, the newest year is used."
        ),
    )