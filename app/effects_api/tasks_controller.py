import uuid
from typing import Annotated

from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import JSONResponse

from app.common.auth.auth import verify_token
from app.effects_api.modules.task_service import (
    TASK_METHODS,
    AnyTask,
    _task_map,
    _task_queue,
)

from ..common.caching.caching_service import cache
from ..common.exceptions.http_exception_wrapper import http_exception
from .dto.development_dto import ContextDevelopmentDTO
from .dto.transformation_effects_dto import TerritoryTransformationDTO

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/{method}", status_code=202)
async def create_task(
    method: str,
    params: Annotated[ContextDevelopmentDTO, Depends(ContextDevelopmentDTO)],
    token: str = Depends(verify_token),
):
    if method not in TASK_METHODS:
        raise http_exception(404, f"method '{method}' is not registered", method)

    task_id = f"{method}_{params.scenario_id}"

    existing = _task_map.get(task_id)
    if existing and existing.status in {"queued", "running"}:
        return {"task_id": task_id, "status": existing.status}

    task = AnyTask(method, params.scenario_id, token, params)
    _task_map[task_id] = task
    await _task_queue.put(task)

    return {"task_id": task_id, "status": "queued"}


@router.get("/{task_id}")
async def task_status(task_id: str):
    task = _task_map.get(task_id)
    if not task:
        raise http_exception(404, "task not found", task_id)
    return await task.to_response()


@router.get("/territory_transformation/{scenario_id}/{service_name}")
async def get_tt_layer(scenario_id: int, service_name: str):
    """
    Вернуть уже посчитанный layer (или 404, если нет).
    """
    cached = cache.load("territory_transformation", scenario_id)
    if not cached:
        raise http_exception(404, "no cached result for this scenario", scenario_id)

    fcoll = cached["data"].get(service_name)
    if not fcoll:
        raise http_exception(404, f"service '{service_name}' not found")

    return JSONResponse(content=fcoll)
