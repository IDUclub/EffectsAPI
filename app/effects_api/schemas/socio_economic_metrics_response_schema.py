
from pydantic import BaseModel


class SocioEconomicMetricsResponseSchema(BaseModel):
    results: dict[str, dict[str, dict[str, int | float]]]
