import json
from typing import Annotated

from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import JSONResponse

from app.common.auth.auth import verify_token

from ..common.exceptions.http_exception_wrapper import http_exception
from ..dependencies import effects_service
from .dto.development_dto import (
    ContextDevelopmentDTO,
    DevelopmentDTO,
)
from .dto.socio_economic_project_dto import SocioEconomicByProjectDTO
from .dto.socio_economic_scenario_dto import SocioEconomicByScenarioDTO
from .dto.transformation_effects_dto import TerritoryTransformationDTO
from .schemas.development_response_schema import DevelopmentResponseSchema
from .schemas.socio_economic_response_schema import SocioEconomicResponseSchema

development_router = APIRouter(prefix="/development", tags=["Effects"])
f_22_router = APIRouter(prefix="/f22", tags=["Effects"])
f_26_router = APIRouter(prefix="/f26", tags=["Effects"])
f_35_router = APIRouter(prefix="/f35", tags=["Effects"])
f_36_router = APIRouter(prefix="/f36", tags=["Effects"])


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


@f_22_router.get(
    "/project_socio_economic_prediction", response_model=SocioEconomicResponseSchema
)
async def get_socio_economic_prediction(
    params: Annotated[SocioEconomicByProjectDTO, Depends(SocioEconomicByProjectDTO)],
    token: str = Depends(verify_token),
) -> SocioEconomicResponseSchema:
    return await effects_service.evaluate_master_plan_by_project(params, token)


@f_22_router.get(
    "/scenario_socio_economic_prediction", response_model=SocioEconomicResponseSchema
)
async def get_socio_economic_prediction(
    params: Annotated[SocioEconomicByScenarioDTO, Depends(SocioEconomicByScenarioDTO)],
    token: str = Depends(verify_token),
) -> SocioEconomicResponseSchema:
    return await effects_service.evaluate_master_plan_by_scenario(params, token)


@f_35_router.get("/territory_transformation")
async def territory_transformation(
    params: Annotated[TerritoryTransformationDTO, Depends(TerritoryTransformationDTO)],
    token: str = Depends(verify_token),
):
    gdf = await effects_service.territory_transformation_scenario_before(token, params)
    gdf = gdf.to_crs(4326)

    geojson_dict = json.loads(gdf.to_json(drop_id=True))
    return JSONResponse(content=geojson_dict)


@f_26_router.get("/values_development")
async def values_development(
    params: Annotated[ContextDevelopmentDTO, Depends(ContextDevelopmentDTO)],
    token: str = Depends(verify_token),
):
    return await effects_service.values_transformation(token, params)


@f_36_router.get("/values_oriented_requirements")
async def values_requirements(
    params: Annotated[TerritoryTransformationDTO, Depends(TerritoryTransformationDTO)],
    token: str = Depends(verify_token),
):
    return await effects_service.values_oriented_requirements(token, params)
