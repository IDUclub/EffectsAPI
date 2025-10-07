import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal

import geopandas as gpd
from fastapi import FastAPI
from loguru import logger

from app.dependencies import effects_service, file_cache

MethodFunc = Callable[[str, Any], "dict[str, Any]"]

TASK_METHODS: dict[str, MethodFunc] = {
    "territory_transformation": effects_service.territory_transformation,
    "values_transformation": effects_service.values_transformation,
    "values_oriented_requirements": effects_service.values_oriented_requirements,
}

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

    def run_sync(self) -> None:
        try:
            logger.info(f"[{self.task_id}] started")
            self.status = "running"

            force = getattr(self.params, "force", False)

            cached = None if force else self.cache.load(self.method, self.scenario_id, self.param_hash)

            def cache_complete(method: str, cached_obj: dict | None) -> bool:
                if not cached_obj:
                    return False
                data = cached_obj.get("data") or {}
                if method == "territory_transformation":
                    return bool(data.get("after"))
                return True

            if not force and cache_complete(self.method, cached):
                logger.info(f"[{self.task_id}] loaded from cache")
                self.result = cached["data"]
                self.status = "done"
                return

            func = TASK_METHODS[self.method]
            raw_data = asyncio.run(func(self.token, self.params))

            def gdf_to_dict(gdf: gpd.GeoDataFrame) -> dict:
                return json.loads(gdf.to_json(drop_id=True))

            if isinstance(raw_data, gpd.GeoDataFrame):
                data_to_cache = gdf_to_dict(raw_data)
            elif isinstance(raw_data, dict):
                data_to_cache = {
                    k: gdf_to_dict(v) if isinstance(v, gpd.GeoDataFrame) else v
                    for k, v in raw_data.items()
                }
            else:
                data_to_cache = raw_data

            self.result = data_to_cache
            self.status = "done"

        except Exception as exc:
            logger.exception(exc)
            self.status = "failed"
            self.error = str(exc)


async def create_task(method: str, token: str, params, task_id: str) -> str:
    norm_params = await effects_service.get_optimal_func_zone_data(params, token)
    params_for_hash = await effects_service.build_hash_params(norm_params, token)
    phash = file_cache.params_hash(params_for_hash)

    task = AnyTask(
        method, norm_params.scenario_id, token, norm_params, phash, file_cache, task_id
    )
    _task_map[task.task_id] = task
    await _task_queue.put(task)
    return task.task_id


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
            await asyncio.to_thread(task.run_sync)
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
    worker.start()
    try:
        yield
    finally:
        await worker.stop()
