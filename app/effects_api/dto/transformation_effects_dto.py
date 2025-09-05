from pydantic import Field

from app.effects_api.dto.development_dto import ContextDevelopmentDTO


class TerritoryTransformationDTO(ContextDevelopmentDTO):
    required_service: str = Field(
        ...,
        examples=["school"],
        description="Service type to get response on",
    )
