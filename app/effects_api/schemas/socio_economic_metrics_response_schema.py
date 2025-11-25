from pydantic import BaseModel, Field


class SocioEconomicMetricsResponseSchema(BaseModel):
    results: dict[str, dict[str, dict[str, int | float | None]]] = Field(
        ..., description="Results of socio economic metrics"
    )
