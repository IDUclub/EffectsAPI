import asyncio
import contextlib
import json
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Callable, Literal

import geopandas as gpd
from fastapi import FastAPI
from loguru import logger

from app.common.caching.caching_service import cache
from app.effects_api.effects_service import effects_service

MethodFunc = Callable[[str, Any], "dict[str, Any]"]

TASK_METHODS: dict[str, MethodFunc] = {
    "territory_transformation": effects_service.territory_transformation_scenario,
}

_task_queue: asyncio.Queue["AnyTask"] = asyncio.Queue()
_task_map: dict[str, "AnyTask"] = {}


class AnyTask:
    def __init__(self, method: str, scenario_id: int, token: str, params: Any):
        self.method = method
        self.scenario_id = scenario_id
        self.token = token
        self.params = params

        self.param_hash = cache.params_hash(params.model_dump())
        self.task_id = f"{method}_{scenario_id}_{self.param_hash}"
        self.status: Literal["queued", "running", "done", "failed"] = "queued"
        self.result: dict | None = None
        self.error: str | None = None

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

            # 1. Пытаемся взять кэш по (method, id, hash)
            cached = cache.load(self.method, self.scenario_id, self.param_hash)
            if cached:
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

            cache.save(
                self.method,
                self.scenario_id,
                self.params.model_dump(),
                data_to_cache,
            )

            self.result = data_to_cache
            self.status = "done"

        except Exception as exc:
            logger.exception(exc)
            self.status = "failed"
            self.error = str(exc)


async def _worker():
    while True:
        task: AnyTask = await _task_queue.get()
        await asyncio.to_thread(task.run_sync)
        _task_queue.task_done()


worker_task: asyncio.Task | None = None


def init_worker(app: FastAPI):
    global worker_task
    worker_task = asyncio.create_task(_worker(), name="any_task_worker")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if worker_task:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
