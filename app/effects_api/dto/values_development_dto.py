from pydantic import Field

from app.effects_api.dto.development_dto import ContextDevelopmentDTO


class ValuesDevelopmentDTO(ContextDevelopmentDTO):
    required_service: str = Field(
        ...,
        examples=["school"],
        description="Service type to get response on",
    )
