import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal, Coroutine

import geopandas as gpd
from fastapi import FastAPI
from loguru import logger

from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import effects_service, file_cache, effects_utils, consumer, producer, config
from app.prometheus.server import start_metrics_server, stop_metrics_server
import time

from app.prometheus.metrics import (
    bind_queue_metrics, get_task_metrics)

MethodFunc = Callable[[str, Any], Coroutine[Any, Any, Any]]
TASK_METRICS = get_task_metrics()

TASK_METHODS: dict[str, MethodFunc] = {
    "territory_transformation": effects_service.territory_transformation,
    "values_transformation": effects_service.values_transformation,
    "values_oriented_requirements": effects_service.values_oriented_requirements,
    "social_economical_metrics": effects_service.evaluate_social_economical_metrics
}


def _cache_complete(method: str, cached_obj: dict | None) -> bool:
    if not cached_obj:
        return False
    data = cached_obj.get("data") or {}
    if method == "territory_transformation":
        if data.get("after"):
            return True
        return bool(data.get("before"))
    return True

_task_queue: asyncio.Queue["AnyTask"] = asyncio.Queue()
_task_map: dict[str, "AnyTask"] = {}


class AnyTask:
    def __init__(
        self,
        method: str,
        scenario_id: int,
        token: str,
        params: Any,
        params_hash: str,
        cache: file_cache,
        task_id: str,
    ):
        self.method = method
        self.scenario_id = scenario_id
        self.token = token
        self.params = params
        self.param_hash = params_hash

        self.status: Literal["queued", "running", "done", "failed"] = "queued"
        self.result: dict | None = None
        self.error: str | None = None
        self.cache = cache
        self.task_id = task_id

    async def to_response(self) -> dict:
        if self.status in {"queued", "running"}:
            return {"status": self.status}
        if self.status == "done":
            return {"status": "done", "result": self.result}
        return {"status": "failed", "error": self.error}

    async def run(self) -> None:
        """
        Run task asynchronously inside event loop.
        """
        start_time = time.perf_counter()
        TASK_METRICS.on_started(self.method)

        try:
            logger.info(f"[{self.task_id}] started")
            self.status = "running"

            force = getattr(self.params, "force", False)
            cached = None if force else self.cache.load(self.method, self.scenario_id, self.param_hash)

            if not force and _cache_complete(self.method, cached):
                logger.info(f"[{self.task_id}] loaded from cache")
                TASK_METRICS.on_cache_hit(self.method)
                self.result = cached["data"]
                self.status = "done"
                return

            func = TASK_METHODS[self.method]
            raw_data = await func(self.token, self.params)

            self.result = self._serialize_result(raw_data)
            self.status = "done"
            TASK_METRICS.on_finished_success(self.method)


        except Exception as exc:
            logger.exception(f"[{self.task_id}] failed")
            self.status = "failed"
            self.error = str(exc)
            TASK_METRICS.on_finished_failed(self.method)

        finally:
            TASK_METRICS.observe_duration(self.method, time.perf_counter() - start_time)

    def _serialize_result(self, raw_data):
        """Serialize GeoDataFrame or dict to json-compatible structure."""
        if isinstance(raw_data, gpd.GeoDataFrame):
            return json.loads(raw_data.to_json(drop_id=True))

        if isinstance(raw_data, dict):
            return {
                k: json.loads(v.to_json(drop_id=True))
                if isinstance(v, gpd.GeoDataFrame)
                else v
                for k, v in raw_data.items()
            }

        return raw_data


async def create_task(
    method: str,
    token: str,
    params,
) -> dict:
    """
    Create (or reuse) an async Effects task.

    Returns:
        dict: { "task_id": str, "status": "queued" | "running" | "done" }
    """
    TASK_METRICS.on_created(method)

    project_based_methods = {"social_economical_metrics", "urbanomy_metrics"}

    if method in project_based_methods:
        owner_id = getattr(params, "project_id", None)

        params_for_hash = {
            "project_id": getattr(params, "project_id", None),
            "regional_scenario_id": getattr(params, "regional_scenario_id", None),
        }

        force = bool(getattr(params, "force", False))

        try:
            phash = file_cache.params_hash(params_for_hash)
        except Exception as e:
            logger.exception("Failed to hash params (project)")
            raise http_exception(500, "Failed to hash task parameters",
                                 _input=params_for_hash, _detail=str(e))

        task_id = f"{method}_{owner_id}_{phash}"

        try:
            cached = None if force else file_cache.load(method, owner_id, phash)
        except Exception as e:
            logger.exception("Cache load failed (project)")
            raise http_exception(500, "Cache load failed",
                                 _input={"method": method, "owner_id": owner_id}, _detail=str(e))

        if not force and _cache_complete(method, cached):
            TASK_METRICS.on_cache_hit(method)
            return {"task_id": task_id, "status": "done"}

        existing = None if force else _task_map.get(task_id)
        if not force and existing and existing.status in {"queued", "running"}:
            return {"task_id": task_id, "status": existing.status}

        task = AnyTask(method, owner_id, token, params, phash, file_cache, task_id)
        _task_map[task_id] = task
        TASK_METRICS.on_enqueued(method)
        await _task_queue.put(task)
        return {"task_id": task_id, "status": "queued"}

    if method == "values_oriented_requirements":
        base_id = await effects_utils._resolve_base_id(token, getattr(params, "scenario_id"))
        logger.info(
            "[Tasks] values_oriented_requirements base_id=%s (requested=%s)",
            base_id, getattr(params, "scenario_id")
        )

        base_params = params.model_copy(update={
            "scenario_id": base_id,
            "proj_func_zone_source": None,
            "proj_func_source_year": None,
            "context_func_zone_source": None,
            "context_func_source_year": None,
        })
        norm_params = await effects_service.get_optimal_func_zone_data(base_params, token)

        params_for_hash = await effects_service.build_hash_params(norm_params, token)
        phash = file_cache.params_hash(params_for_hash)
        owner_id = base_id
        task_id = f"{method}_{owner_id}_{phash}"

        cached = file_cache.load(method, owner_id, phash)
        if cached and "data" in cached and "result" in cached["data"]:
            logger.info("[Tasks] Cache hit for values_oriented_requirements -> DONE")
            TASK_METRICS.on_cache_hit(method)
            return {"task_id": task_id, "status": "done"}

        task = AnyTask(method, owner_id, token, norm_params, phash, file_cache, task_id)
        if task.task_id in _task_map:
            return {"task_id": task.task_id, "status": "running"}
        _task_map[task.task_id] = task
        TASK_METRICS.on_enqueued(method)
        await _task_queue.put(task)
        return {"task_id": task.task_id, "status": "queued"}

    norm_params = await effects_service.get_optimal_func_zone_data(params, token)
    params_for_hash = await effects_service.build_hash_params(norm_params, token)
    phash = file_cache.params_hash(params_for_hash)
    owner_id = norm_params.scenario_id
    task_id = f"{method}_{owner_id}_{phash}"

    cached = file_cache.load(method, owner_id, phash)
    if cached and "data" in cached:
        TASK_METRICS.on_cache_hit(method)
        return {"task_id": task_id, "status": "done"}

    task = AnyTask(method, owner_id, token, norm_params, phash, file_cache, task_id)
    if task.task_id in _task_map:
        return {"task_id": task.task_id, "status": "running"}
    _task_map[task.task_id] = task
    TASK_METRICS.on_enqueued(method)
    await _task_queue.put(task)
    return {"task_id": task.task_id, "status": "queued"}


async def _worker():
    while True:
        task: AnyTask = await _task_queue.get()
        await asyncio.to_thread(task.run_sync)
        _task_queue.task_done()


worker_task: asyncio.Task | None = None


class Worker:
    def __init__(self):
        self.is_alive = True
        self.task: asyncio.Task | None = None

    async def run(self):
        while self.is_alive:
            task: AnyTask = await _task_queue.get()
            await task.run()
            _task_queue.task_done()

    def start(self):
        self.task = asyncio.create_task(self.run(), name="any_task_worker")

    async def stop(self):
        self.is_alive = False
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


worker = Worker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_metrics_server(int(config.get("PROMETHEUS_PORT")))
    bind_queue_metrics(_task_queue)
    worker.start()
    await producer.start()
    await consumer.start(["scenario.events"])
    try:
        yield
    finally:
        await consumer.stop()
        await producer.stop()
        await worker.stop()
        stop_metrics_server()
