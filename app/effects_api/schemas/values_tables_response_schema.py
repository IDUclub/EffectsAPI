from typing import Dict, List

from pydantic import BaseModel, Field


class ValuesOrientedResponseTablesSchema(BaseModel):
    base_scenario_id: int = Field(..., description="Id of the base scenario")
    values_table: Dict[str, Dict[str, str | float]] = Field(
        ..., description="Values table for the base scenario"
    )
    services_type_deficit: List[Dict[str, str | float | List[str]]] = Field(
        ..., description="Services type deficit for the base scenario"
    )
