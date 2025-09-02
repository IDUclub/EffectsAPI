from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_serializer

from app.common.exceptions.http_exception_wrapper import http_exception
from app.effects_api.dto.development_dto import ContextDevelopmentDTO, DevelopmentDTO

from .output_maps import pred_columns_names_map, soc_economy_pred_name_map


class SocioEconomicParams(BaseModel):
    pred: int = Field(..., description="Prediction column name")
    lower: int = Field(..., description="Lower prediction column name")
    upper: float = Field(..., description="Upper prediction column name")
    is_interval: bool = Field(..., description="Is interval prediction column name")

    @field_validator("upper", mode="after")
    @classmethod
    def validate_upper(cls, v: float) -> float:
        return round(v, 2)

    @model_serializer
    def serialize_model(self):
        return {
            pred_columns_names_map[str(field)]: getattr(self, field)
            for field in self.model_fields
        }


class SocioEconomicSchema(BaseModel):
    socio_economic_prediction: dict[str, SocioEconomicParams]

    @field_validator("socio_economic_prediction", mode="after")
    @classmethod
    def rename_attributes(cls, value: dict[str, SocioEconomicParams]):

        try:
            return {
                (
                    k
                    if k not in pred_columns_names_map.keys()
                    else soc_economy_pred_name_map[k]
                ): v
                for k, v in value.items()
            }
        except KeyError as key_e:
            raise http_exception(
                500,
                "Could not rename socio economic prediction attributes",
                _input=list(value.keys()),
                _detail={"error": repr(key_e)},
            ) from key_e
        except Exception as e:
            raise http_exception(
                500,
                "Error during output validation",
                _input=list(value.keys()),
                _detail={"error": repr(e)},
            ) from e


class SocioEconomicResponseSchema(SocioEconomicSchema):
    """
    DTO Class for socio-economic response
    Attributes:
        socio_economic_prediction (dict[str, SocioEconomicParams]): where SocioEconomicParams is class containing fields
        (pred: int, lower: int, upper: float, is_interval: bool)
        split_prediction (Optional[list[SocioEconomicSchema]]): optional list of context predictions as list
        of SocioEconomicSchema
        params_data (DevelopmentDTO | ContextDevelopmentDTO):
    """

    split_prediction: Optional[dict[int, SocioEconomicSchema]]
    params_data: DevelopmentDTO | ContextDevelopmentDTO
