from typing_extensions import Self

from pydantic import BaseModel, Field, model_validator

from app.effects_api.dto.development_dto import DevelopmentDTO, ContextDevelopmentDTO


class DevelopmentResponseSchema(BaseModel):
    """
    Schema for the response of the development endpoint.
    """

    site_area: list[float]
    fsi: list[float]
    gsi: list[float]
    mxi: list[float]
    build_floor_area: list[float]
    footprint_area: list[float]
    living_area: list[float]
    non_living_area: list[float]
    population: list[float]
    params_data: DevelopmentDTO | ContextDevelopmentDTO

    @model_validator(mode="after")
    def round_floats(self) -> Self:
        for field, value in self.model_fields.items():
            if self.model_fields[field].annotation == list[float]:
                rounded = [round(i, 2) for i in getattr(self, field)]
                setattr(self, field, rounded)
        return self
