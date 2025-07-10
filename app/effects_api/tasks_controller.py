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
from .effects_service import effects_service

router = APIRouter(prefix="/tasks", tags=["tasks"])


async def _with_defaults(
    dto: ContextDevelopmentDTO, token: str
) -> ContextDevelopmentDTO:
    return await effects_service.get_optimal_func_zone_data(dto, token)


@router.post("/{method}", status_code=202)
async def create_task(
    method: str,
    params: Annotated[ContextDevelopmentDTO, Depends()],
    token: str = Depends(verify_token),
):
    if method not in TASK_METHODS:
        raise http_exception(404, f"method '{method}' is not registered", method)

    params_filled = await effects_service.get_optimal_func_zone_data(params, token)
    phash = cache.params_hash(params_filled.model_dump())
    task_id = f"{method}_{params_filled.scenario_id}_{phash}"

    if cache.load(method, params_filled.scenario_id, phash):
        return {"task_id": task_id, "status": "done"}

    existing = _task_map.get(task_id)
    if existing and existing.status in {"queued", "running"}:
        return {"task_id": task_id, "status": existing.status}

    task = AnyTask(method, params_filled.scenario_id, token, params_filled)
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
    cached = cache.load_latest("territory_transformation", scenario_id)
    if not cached:
        raise http_exception(404, "no saved result for this scenario", scenario_id)

    data: dict = cached["data"]

    if "after" not in data:
        fc = data["before"].get(service_name)
        if not fc:
            raise http_exception(404, f"service '{service_name}' not found")
        return JSONResponse(content=fc)

    fc_before = data["before"].get(service_name)
    fc_after = data["after"].get(service_name)
    if not (fc_before and fc_after):
        raise http_exception(404, f"service '{service_name}' not found")

    return JSONResponse(content={"before": fc_before, "after": fc_after})
