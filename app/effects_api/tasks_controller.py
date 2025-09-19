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

from ..common.exceptions.http_exception_wrapper import http_exception
from ..dependencies import effects_service, file_cache
from .dto.development_dto import ContextDevelopmentDTO
from .modules.service_type_service import get_services_with_ids_from_layer

router = APIRouter(prefix="/tasks", tags=["tasks"])


async def _with_defaults(
    dto: ContextDevelopmentDTO, token: str
) -> ContextDevelopmentDTO:
    return await effects_service.get_optimal_func_zone_data(dto, token)


@router.get("/methods")
async def get_methods():
    """router for getting method names available for tasks creation"""
    return list(TASK_METHODS.keys())


@router.post("/{method}", status_code=202)
async def create_task(
    method: str,
    params: Annotated[ContextDevelopmentDTO, Depends()],
    token: str = Depends(verify_token),
):
    if method not in TASK_METHODS:
        raise http_exception(404, f"method '{method}' is not registered", method)

    params_filled = await effects_service.get_optimal_func_zone_data(params, token)

    params_for_hash = await effects_service.build_hash_params(params_filled, token)
    phash = file_cache.params_hash(params_for_hash)

    task_id = f"{method}_{params_filled.scenario_id}_{phash}"

    if file_cache.load(method, params_filled.scenario_id, phash):
        return {"task_id": task_id, "status": "done"}

    existing = _task_map.get(task_id)
    if existing and existing.status in {"queued", "running"}:
        return {"task_id": task_id, "status": existing.status}

    task = AnyTask(
        method,
        params_filled.scenario_id,
        token,
        params_filled,
        params_for_hash,
        file_cache,
        task_id,
    )
    _task_map[task_id] = task
    await _task_queue.put(task)

    return {"task_id": task_id, "status": "queued"}


@router.get("/status/{task_id}")
async def task_status(task_id: str):
    method, scenario_id, phash = file_cache.parse_task_id(task_id)
    if method and scenario_id is not None and phash:
        try:
            cached = file_cache.load(method, scenario_id, phash)
            if cached:
                return {"task_id": task_id, "status": "done"}
        except Exception:
            pass

    task = _task_map.get(task_id)
    if task:
        return {
            "task_id": task_id,
            "status": getattr(task, "status", "unknown"),
            **(
                {"error": str(task.error)}
                if getattr(task, "status", None) == "failed"
                and getattr(task, "error", None)
                else {}
            ),
        }

    raise http_exception(404, "task not found", task_id)


@router.get("/get_service_types")
async def get_service_types(
    scenario_id: int,
    method: str = "territory_transformation",
):
    return await get_services_with_ids_from_layer(scenario_id, method, file_cache)


@router.get("/territory_transformation/{scenario_id}/{service_name}")
async def get_territory_transformation_layer(scenario_id: int, service_name: str):
    cached = file_cache.load_latest("territory_transformation", scenario_id)
    if not cached:
        raise http_exception(404, "no saved result for this scenario", scenario_id)

    data: dict = cached["data"]

    if "after" not in data or not data.get("after"):
        fc = data.get("before", {}).get(service_name)
        if not fc:
            raise http_exception(404, f"service '{service_name}' not found")
        return JSONResponse(content=fc)

    before_dict = data.get("before", {}) or {}
    after_dict = data.get("after", {}) or {}

    fc_before = before_dict.get(service_name)
    fc_after = after_dict.get(service_name)

    provision_before = before_dict.get("provision_total_before")
    provision_after = after_dict.get("provision_total_after")

    if fc_before and fc_after:
        return JSONResponse(
            content={
                "before": fc_before,
                "after": fc_after,
                "provision_total_before": provision_before,
                "provision_total_after": provision_after,
            }
        )

    if fc_before and not fc_after:
        return JSONResponse(
            content={"before": fc_before, "provision_total_before": provision_before}
        )

    if fc_after and not fc_before:
        return JSONResponse(
            content={"after": fc_after, "provision_total_after": provision_after}
        )

    raise http_exception(404, f"service '{service_name}' not found")


@router.get("/values_oriented_requirements/{scenario_id}/{service_name}")
async def get_values_oriented_requirements_layer(scenario_id: int, service_name: str):
    cached = file_cache.load_latest("values_oriented_requirements", scenario_id)
    if not cached:
        raise http_exception(404, "no saved result for this scenario", scenario_id)

    data: dict = cached["data"]

    fc_provision = data["provision"].get(service_name)
    values_dict = data["result"]
    if not (fc_provision and values_dict):
        raise http_exception(404, f"service '{service_name}' not found")

    return JSONResponse(content={"geojson": fc_provision, "values_table": values_dict})


@router.get("/get_from_cache/{method_name}/{scenario_id}")
async def get_layer(scenario_id: int, method_name: str):
    cached = file_cache.load_latest(method_name, scenario_id)
    if not cached:
        raise http_exception(404, "no saved result for this scenario", scenario_id)

    data: dict = cached["data"]
    return JSONResponse(content=data)
