from typing import Annotated

from fastapi import APIRouter
from fastapi.params import Depends

from app.common.auth.auth import verify_token
from .dto.development_dto import DevelopmentDTO, ContextDevelopmentDTO

development_router = APIRouter(prefix='/redevelopment', tags=['Effects'])

#TODO describe output schemas
@development_router.get("/project_redevelopment")
async def get_project_redevelopment(
        params: Annotated[DevelopmentDTO, Depends(DevelopmentDTO)],
        token: str = Depends(verify_token),
):
    pass

@development_router.et("/context_redevelopment")
async def get_context_redevelopment(
        params: Annotated[ContextDevelopmentDTO, Depends(ContextDevelopmentDTO)],
        token: str = Depends(verify_token),
):
    pass
