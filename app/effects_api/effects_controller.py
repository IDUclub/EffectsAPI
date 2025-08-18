import json
from typing import Annotated

from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import JSONResponse

from app.common.auth.auth import verify_token

from .dto.development_dto import (
    ContextDevelopmentDTO,
    DevelopmentDTO,
    SocioEconomicPredictionDTO,
)
from .dto.transformation_effects_dto import TerritoryTransformationDTO
from .dto.values_development_dto import ValuesDevelopmentDTO
from .effects_service import effects_service
from .schemas.development_response_schema import DevelopmentResponseSchema
from .schemas.socio_economic_response_schema import SocioEconomicResponseSchema

development_router = APIRouter(prefix="/development", tags=["Effects"])


@development_router.get(
    "/project_development", response_model=DevelopmentResponseSchema
)
async def get_project_development(
    params: Annotated[DevelopmentDTO, Depends(DevelopmentDTO)],
    token: str = Depends(verify_token),
) -> DevelopmentResponseSchema:
    return await effects_service.calc_project_development(token, params)


@development_router.get(
    "/context_development", response_model=DevelopmentResponseSchema
)
async def get_context_development(
    params: Annotated[ContextDevelopmentDTO, Depends(ContextDevelopmentDTO)],
    token: str = Depends(verify_token),
) -> DevelopmentResponseSchema:
    return await effects_service.calc_context_development(token, params)


@development_router.get(
    "/socio_economic_prediction", response_model=SocioEconomicResponseSchema
)
async def get_socio_economic_prediction(
    params: Annotated[SocioEconomicPredictionDTO, Depends(SocioEconomicPredictionDTO)],
    token: str = Depends(verify_token),
) -> SocioEconomicResponseSchema:
    return await effects_service.evaluate_master_plan(params, token)


# @development_router.get("/F_35")
# async def territory_transformation(
#     params: Annotated[TerritoryTransformationDTO, Depends(TerritoryTransformationDTO)],
#     token: str = Depends(verify_token),
# ):
#     return await effects_service.territory_transformation_scenario(token, params)


@development_router.get("/territory_transformation")
async def territory_transformation(
    params: Annotated[TerritoryTransformationDTO, Depends(TerritoryTransformationDTO)],
    token: str = Depends(verify_token),
):
    gdf = await effects_service.territory_transformation_scenario_before(token, params)
    gdf = gdf.to_crs(4326)

    geojson_dict = json.loads(gdf.to_json(drop_id=True))
    return JSONResponse(content=geojson_dict)

@development_router.get("/values_development")
async def values_development(
    params: Annotated[ValuesDevelopmentDTO, Depends(ValuesDevelopmentDTO)],
    token: str = Depends(verify_token)
):
    return await effects_service.values_transformation(token, params)
