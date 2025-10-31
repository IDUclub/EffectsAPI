import asyncio
from typing import Annotated, Union

from fastapi import APIRouter
from fastapi.params import Depends

from app.common.auth.auth import verify_token
from app.effects_api.modules.task_service import (
    TASK_METHODS,
    AnyTask,
    _task_map,
    _task_queue,
)
from .dto.socio_economic_project_dto import SocioEconomicByProjectDTO
from .schemas.service_types_response_schema import ServiceTypesResponse, ValuesServiceTypesResponse
from .schemas.socio_economic_metrics_response_schema import SocioEconomicMetricsResponseSchema
from .schemas.territory_transformation_response_schema import TerritoryTransformationLayerResponse, \
    TerritoryTransformationResponseTablesSchema
from .schemas.values_oriented_response_schema import ValuesOrientedResponseSchema
from .schemas.values_tables_response_schema import ValuesOrientedResponseTablesSchema
from .schemas.values_transformation_response_schema import ValuesTransformationSchema

from ..common.exceptions.http_exception_wrapper import http_exception
from ..dependencies import effects_service, effects_utils, file_cache, urban_api_client
from .dto.development_dto import ContextDevelopmentDTO
from .modules.service_type_service import get_services_with_ids_from_layer

router = APIRouter(prefix="/tasks", tags=["tasks"])

_locks: dict[str, asyncio.Lock] = {}


#TODO continue response schemas

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


@router.get("/methods", summary="List available task methods",
            description=(
                "Returns the current list of task method names that can be scheduled via this API.\n\n"
                "- `territory_transformation` — F 35 scenario-based, create with `POST /tasks/{method}`\n"
                "- `values_transformation` — F 26 scenario-based, create with `POST /tasks/{method}`\n"
                "- `values_oriented_requirements` — F 36 scenario-based, create with `POST /tasks/{method}`\n"
                "- `social_economical_metrics` — F22 project-based, create with `POST /tasks/project/{method}`"
            ))
async def get_methods():
    return list(TASK_METHODS.keys())


@router.post("/{method}", status_code=202,
             summary="Create scenario-based task",
             description=(
                 "Queues an asynchronous **scenario-based** task.\n\n"
                 "**Caching behavior**: if `force=false` and a complete cached result exists "
                 "for the computed parameter hash, the endpoint returns `status=done` immediately. "
                 "Otherwise a task is queued and `status=queued` is returned.\n\n"
                 "**Response statuses**:\n"
                 "- `queued`: task was enqueued successfully\n"
                 "- `running`: a task with the same id is already being processed\n"
                 "- `done`: cached result is available\n"
                 "- `failed`: check `GET /tasks/status/{task_id}` for error details\n\n"
                 "**Task id format**: `{method}_{scenario_id}_{phash}`"
             ))
async def create_scenario_task(
    method: str,
    params: Annotated[ContextDevelopmentDTO, Depends()],
    token: str = Depends(verify_token),
):
    """Roter for task creation"""
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

@router.post("/project/{method}", status_code=202,
             summary="Create project-based task",
             description=(
                 "Queues an asynchronous **project-level** task. Currently supported: "
                 "`social_economical_metrics`.\n\n"
                 "**Hash parameters**: `{project_id, regional_scenario_id, territory_ids}`.\n"
                 "**Caching behavior**: if `force=false` and a complete cached result exists, "
                 "for the computed parameter hash, the endpoint returns `status=done` immediately. "
                 "Otherwise a task is queued and `status=queued` is returned.\n\n"
                 "**Response statuses**:\n"
                 "- `queued`: task was enqueued successfully\n"
                 "- `running`: a task with the same id is already being processed\n"
                 "- `done`: cached result is available\n"
                 "- `failed`: check `GET /tasks/status/{task_id}` for error details\n\n"
                 "**Task id format**: `{method}_{project_id}_{phash}`"
             ))
async def create_project_task(
    method: str,
    params: Annotated[SocioEconomicByProjectDTO, Depends()],
    token: Annotated[str, Depends(verify_token)]
):
    """
    separate endpoint for project-based tasks (e.g., socio_economics).
    """
    if method not in {"social_economical_metrics"}:
        raise http_exception(400, f"method '{method}' is not project-based", method)

    project_id = params.project_id
    regional_id = params.regional_scenario_id

    params_for_hash = {
        "project_id": project_id,
        "regional_scenario_id": regional_id,
        "territory_ids": getattr(params, "territory_ids", []),
    }
    phash = file_cache.params_hash(params_for_hash)
    task_id = f"{method}_{project_id}_{phash}"

    force = getattr(params, "force", False)
    cached = None if force else file_cache.load(method, project_id, phash)
    if not force and _cache_complete(method, cached):
        return {"task_id": task_id, "status": "done"}

    existing = None if force else _task_map.get(task_id)
    if not force and existing and existing.status in {"queued", "running"}:
        return {"task_id": task_id, "status": existing.status}

    task = AnyTask(
        method,
        project_id,
        token,
        params,
        phash,
        file_cache,
        task_id,
    )
    _task_map[task_id] = task
    await _task_queue.put(task)

    return {"task_id": task_id, "status": "queued"}


@router.get("/status/{task_id}",
            summary="Get task status",
            description=(
                "Returns current status for a task id.\n\n"
                "**Statuses**:\n"
                "- `queued`: waiting in queue\n"
                "- `running`: being processed\n"
                "- `done`: cached (final) result exists\n"
                "- `failed`: task failed, `error` field may be present\n"
                "- `unknown`: task is tracked but status cannot be resolved\n\n"
                "If the cache already contains a complete result for the `task_id`, "
                "the endpoint responds with `status=done`."
            ))
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


@router.get(
    "/get_service_types",
    summary="List service types",
    response_model=Union[ServiceTypesResponse, ValuesServiceTypesResponse],
    description=(
                 "Returns service type identifiers available for a given `scenario_id` and `method` "
                 "from the cached layer. Intended to help clients discover which services can be requested."
                 "For 'territory_transformation' method 'before' and 'after' keys with services are returned"
                 "For  'values_oriented_requirements' only 'services' key with services is returned"
                ),
    response_model_exclude_none=True,
)
async def get_service_types(
    scenario_id: int,
    method: str = "territory_transformation",
    token: str = Depends(verify_token),
):
    """Return service types depending on the method."""
    if method == "territory_transformation":
        data = await get_services_with_ids_from_layer(
            scenario_id, method, file_cache, effects_utils, token=token
        )
        return ServiceTypesResponse(before=data["before"], after=data.get("after", []))

    if method == "values_oriented_requirements":
        services = await get_services_with_ids_from_layer(
            scenario_id, method, file_cache, effects_utils, token=token
        )
        return ValuesServiceTypesResponse(
            services=services.get("services", [])
        )
    raise http_exception(400, f"Unsupported method", f"{method}")


@router.get("/territory_transformation/{scenario_id}/{service_name}",
            summary="Get territory transformation layer by service",
            description=(
                "Fetches a GeoJSON layer for a specific `service_name` from the cached "
                "`territory_transformation` result.\n\n"
                "**Responses**:\n"
                "- When both versions exist: returns `{ before, after, provision_total_before, provision_total_after }`\n"
                "- When only `before` exists: returns `{ before, provision_total_before }`\n"
                "- When only `after` exists: returns `{ after, provision_total_after }`"
            ),
            response_model=TerritoryTransformationLayerResponse,
            )
async def get_territory_transformation_layer(scenario_id: int, service_name: str):
    cached = file_cache.load_latest("territory_transformation", scenario_id)
    if not cached:
        raise http_exception(404, "no saved result for this scenario", scenario_id)

    data: dict = cached["data"]

    if "after" not in data or not data.get("after"):
        fc = data.get("before", {}).get(service_name)
        if not fc:
            raise http_exception(404, f"service '{service_name}' not found")
        return TerritoryTransformationLayerResponse(before= fc)

    before_dict = data.get("before", {}) or {}
    after_dict = data.get("after", {}) or {}

    fc_before = before_dict.get(service_name)
    fc_after = after_dict.get(service_name)

    provision_before = before_dict.get("provision_total_before")
    provision_after = after_dict.get("provision_total_after")

    if fc_before and fc_after:
        return TerritoryTransformationLayerResponse(
            before =  fc_before,
            after = fc_after,
            provision_total_before = provision_before,
            provision_total_after = provision_after,
        )

    if fc_before and not fc_after:
        return TerritoryTransformationLayerResponse(
            before = fc_before, provision_total_before = provision_before)

    if fc_after and not fc_before:
        return TerritoryTransformationLayerResponse(
            after= fc_after, provision_total_after = provision_after
        )

    raise http_exception(404, f"service '{service_name}' not found")


@router.get("/values_oriented_requirements/{scenario_id}/{service_name}",
            summary="Get Values-Oriented Requirements layer",
            description=(
                "Returns the GeoJSON layer and values table for a `service_name`, computed for the "
                "**base scenario** of the provided `scenario_id`.\n\n"
                "Rejects the request if the cached base result is stale compared to the base scenario metadata."
            ),
            response_model=ValuesOrientedResponseSchema)
async def get_values_oriented_requirements_layer(
    scenario_id: int,
    service_name: str,
    token: str = Depends(verify_token),
):
    base_id = await effects_utils._resolve_base_id(token, scenario_id)

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
    values_table = data.get("social_values_table")

    if not prov:
        raise http_exception(
            404, f"service '{service_name}' not found in base scenario {base_id}"
        )

    return ValuesOrientedResponseSchema(
            base_scenario_id= base_id,
            geojson= prov,
            values_table= values_dict,
            services_type_deficit= values_table,
    )


@router.get("/values_oriented_requirements_table/{scenario_id}",
            summary="Get Values-Oriented Requirements tables",
            description=(
                "Returns the values table and service-type deficit table for the **base scenario** "
                "of the provided `scenario_id`."
            ),
            response_model=ValuesOrientedResponseTablesSchema
            )
async def get_values_oriented_requirements_table(
    scenario_id: int,
    token: str = Depends(verify_token),
):
    base_id = await effects_utils.resolve_base_id(token, scenario_id)

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
    values_table = data.get("social_values_table")

    return ValuesOrientedResponseTablesSchema(
            base_scenario_id = base_id,
            values_table = values_dict,
            services_type_deficit = values_table,
    )


@router.get("/get_from_cache/{method_name}/{project_scenario_id}",
            summary="Get raw cached data by method and owner id",
            description=(
                "Reads the latest cached JSON payload for a given `method_name` and owner id. "
                "For scenario-based methods the owner is a **scenario id**; for project-based "
                "methods the owner is a **project id**."
            ),
            response_model=Union[ValuesTransformationSchema, SocioEconomicMetricsResponseSchema])
async def get_layer(project_scenario_id: int, method_name: str):
    cached = file_cache.load_latest(method_name, project_scenario_id)
    if not cached:
        raise http_exception(404, "no saved result for this scenario", project_scenario_id)

    data = cached["data"]

    if method_name == "values_transformation":
        return ValuesTransformationSchema(geojson=data)

    if method_name == "social_economical_metrics":
        data = cached["data"]["results"]
        return SocioEconomicMetricsResponseSchema(results=data)

    else:
        raise http_exception(400, "Method not implemented", method_name, "Allowed methods: values_transformation, social_economical_metrics")


@router.get("/get_provisions/{scenario_id}",
            summary="Get total provision values",
            description=(
                "Returns total provision values from the cached `territory_transformation` result for "
                "the specified `scenario_id`. Depending on availability, the response contains:\n"
                "- `provision_total_before` and `provision_total_after`, or\n"
                "- only one of them if the other is not present."
            ),
            response_model=TerritoryTransformationResponseTablesSchema)
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
        return TerritoryTransformationResponseTablesSchema(
                provision_total_before = provision_before,
                provision_total_after= provision_after,
        )

    if provision_before and not provision_after:
        return TerritoryTransformationResponseTablesSchema(provision_total_before = provision_before)

    if provision_after and not provision_before:
        return TerritoryTransformationResponseTablesSchema(provision_total_after= provision_after)

    raise http_exception(404, f"Result for scenario ID{scenario_id} not found")
