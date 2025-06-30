from pydantic import BaseModel, field_validator

from app.effects_api.dto.development_dto import (ContextDevelopmentDTO,
                                                 DevelopmentDTO)


class SocioEconomicParams(BaseModel):

    pred: int
    lower: int
    upper: float
    is_interval: bool

    @field_validator("upper", mode="after")
    @classmethod
    def validate_upper(cls, v: float) -> float:
        return round(v, 2)


class SocioEconomicResponseSchema(BaseModel):

    socio_economic_prediction: dict[str, SocioEconomicParams]
    params_data: DevelopmentDTO | ContextDevelopmentDTO
