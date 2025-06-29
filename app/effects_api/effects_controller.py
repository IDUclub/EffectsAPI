from typing import Annotated

from fastapi import APIRouter
from fastapi.params import Depends

from app.common.auth.auth import verify_token
from app.common.exceptions.controller_exception_handler import handle_controller_exception
from .dto.development_dto import DevelopmentDTO, ContextDevelopmentDTO
from .effects_service import effects_service
from .schemas.development_response_schema import DevelopmentResponseSchema
from .schemas.socio_economic_response_schema import SocioEconomicResponseSchema


development_router = APIRouter(prefix='/redevelopment', tags=['Effects'])


@development_router.get("/project_redevelopment", response_model=DevelopmentResponseSchema)
async def get_project_redevelopment(
        params: Annotated[DevelopmentDTO, Depends(DevelopmentDTO)],
        token: str = Depends(verify_token),
) -> DevelopmentResponseSchema:

    return await handle_controller_exception(effects_service.calc_project_development, token=token, params=params)


@development_router.get("/context_redevelopment", response_model=DevelopmentResponseSchema)
async def get_context_redevelopment(
        params: Annotated[ContextDevelopmentDTO, Depends(ContextDevelopmentDTO)],
        token: str = Depends(verify_token),
) -> DevelopmentResponseSchema:

    return await handle_controller_exception(effects_service.calc_context_development, token=token, params=params)


@development_router.get("/socio_economic_prediction", response_model=SocioEconomicResponseSchema)
async def get_socio_economic_prediction(
        params: Annotated[ContextDevelopmentDTO, Depends(ContextDevelopmentDTO)],
        token: str = Depends(verify_token),
) -> SocioEconomicResponseSchema:

    return await handle_controller_exception(
        effects_service.evaluate_master_plan, token=token, params=params
    )
