from pydantic import Field

from app.effects_api.dto.development_dto import ContextDevelopmentDTO


class SocioEconomicByScenarioDTO(ContextDevelopmentDTO):

    split: bool = Field(
        default=False,
        examples=[False, True],
        description="If split will return additional evaluation for each context mo",
    )
