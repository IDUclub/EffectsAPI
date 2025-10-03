import asyncio
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
from ..common.utils.ids_convertation import _resolve_base_id
from ..dependencies import effects_service, file_cache, urban_api_client
from .dto.development_dto import ContextDevelopmentDTO
from .modules.service_type_service import get_services_with_ids_from_layer

router = APIRouter(prefix="/tasks", tags=["tasks"])

_locks: dict[str, asyncio.Lock] = {}


def _get_lock(key: str) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock


async def _with_defaults(
    dto: ContextDevelopmentDTO, token: str
) -> ContextDevelopmentDTO:
    return await effects_service.get_optimal_func_zone_data(dto, token)


def _is_fc(x: dict) -> bool:
    return (
        isinstance(x, dict)
        and x.get("type") == "FeatureCollection"
        and isinstance(x.get("features"), list)
    )


def _section_ready(sec: dict | None) -> bool:
    return isinstance(sec, dict) and any(_is_fc(v) for v in sec.values())


def _cache_complete(method: str, cached: dict | None) -> bool:
    if not cached:
        return False
    data = cached.get("data") or {}
    if method == "territory_transformation":
        return _section_ready(data.get("before")) and _section_ready(data.get("after"))
    return True


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

    coarse_key = f"{method}:{params.scenario_id}"
    lock = _get_lock(coarse_key)

    async with lock:
        params_filled = await effects_service.get_optimal_func_zone_data(params, token)
        params_for_hash = await effects_service.build_hash_params(params_filled, token)
        phash = file_cache.params_hash(params_for_hash)

        task_id = f"{method}_{params_filled.scenario_id}_{phash}"

        force = getattr(params, "force", False)

        cached = (
            None if force else file_cache.load(method, params_filled.scenario_id, phash)
        )
        if not force and _cache_complete(method, cached):
            return {"task_id": task_id, "status": "done"}

        existing = None if force else _task_map.get(task_id)
        if not force and existing and existing.status in {"queued", "running"}:
            return {"task_id": task_id, "status": existing.status}

        task = AnyTask(
            method,
            params_filled.scenario_id,
            token,
            params_filled,
            phash,
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
            if _cache_complete(method, cached):
                return {"task_id": task_id, "status": "done"}
            if cached:
                return {"task_id": task_id, "status": "running"}
        except Exception:
            pass

    task = _task_map.get(task_id)
    if task:
        payload = {
            "task_id": task_id,
            "status": getattr(task, "status", "unknown"),
        }
        if getattr(task, "status", None) == "failed" and getattr(task, "error", None):
            payload["error"] = str(task.error)
        return payload

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
async def get_values_oriented_requirements_layer(
    scenario_id: int,
    service_name: str,
    token: str = Depends(verify_token),
):
    base_id = await _resolve_base_id(urban_api_client, token, scenario_id)

    cached = file_cache.load_latest("values_oriented_requirements", base_id)
    if not cached:
        raise http_exception(
            404, f"no saved result for base scenario {base_id}", base_id
        )

    info_base = await urban_api_client.get_scenario_info(base_id, token)
    if cached.get("meta", {}).get("scenario_updated_at") != info_base.get("updated_at"):
        raise http_exception(
            404, f"stale cache for base scenario {base_id}, recompute required", base_id
        )

    data: dict = cached.get("data", {})
    prov = (data.get("provision") or {}).get(service_name)
    values_dict = data.get("result")

    if not prov:
        raise http_exception(
            404, f"service '{service_name}' not found in base scenario {base_id}"
        )

    return JSONResponse(
        content={
            "base_scenario_id": base_id,
            "geojson": prov,
            "values_table": values_dict,
        }
    )


@router.get("/values_oriented_requirements_table/{scenario_id}")
async def get_values_oriented_requirements_table(
    scenario_id: int,
    token: str = Depends(verify_token),
):
    base_id = await _resolve_base_id(urban_api_client, token, scenario_id)

    cached = file_cache.load_latest("values_oriented_requirements", base_id)
    if not cached:
        raise http_exception(
            404, f"no saved result for base scenario {base_id}", base_id
        )

    info_base = await urban_api_client.get_scenario_info(base_id, token)
    if cached.get("meta", {}).get("scenario_updated_at") != info_base.get("updated_at"):
        raise http_exception(
            404, f"stale cache for base scenario {base_id}, recompute required", base_id
        )

    data: dict = cached.get("data", {})
    values_dict = data.get("result")

    return JSONResponse(
        content={
            "base_scenario_id": base_id,
            "values_table": values_dict,
        }
    )


@router.get("/get_from_cache/{method_name}/{scenario_id}")
async def get_layer(scenario_id: int, method_name: str):
    cached = file_cache.load_latest(method_name, scenario_id)
    if not cached:
        raise http_exception(404, "no saved result for this scenario", scenario_id)

    data: dict = cached["data"]
    return JSONResponse(content=data)


@router.get("/get_provisions/{scenario_id}")
async def get_total_provisions(scenario_id: int):
    cached = file_cache.load_latest("territory_transformation", scenario_id)
    if not cached:
        raise http_exception(404, "no saved result for this scenario", scenario_id)

    data: dict = cached["data"]

    before_dict = data.get("before", {}) or {}
    after_dict = data.get("after", {}) or {}

    provision_before = before_dict.get("provision_total_before")
    provision_after = after_dict.get("provision_total_after")

    if provision_before and provision_after:
        return JSONResponse(
            content={
                "provision_total_before": provision_before,
                "provision_total_after": provision_after,
            }
        )

    if provision_before and not provision_after:
        return JSONResponse(content={"provision_total_before": provision_before})

    if provision_after and not provision_before:
        return JSONResponse(content={"provision_total_after": provision_after})

    raise http_exception(404, f"Result for scenario ID{scenario_id} not found")
