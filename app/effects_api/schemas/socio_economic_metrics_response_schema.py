
from pydantic import BaseModel


class SocioEconomicMetricsResponseSchema(BaseModel):
    results: dict[str, list[dict[str, int | float]]]