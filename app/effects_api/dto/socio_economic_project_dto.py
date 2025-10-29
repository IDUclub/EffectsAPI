from pydantic import BaseModel, Field


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

    force: bool = Field(
        default=False, description="flag for recalculating the scenario"
    )
