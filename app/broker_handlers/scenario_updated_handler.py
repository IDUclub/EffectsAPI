from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from confluent_kafka import Message
from loguru import logger
from otteroad import BaseMessageHandler, KafkaProducerClient
from otteroad.consumer.handlers.base import EventT
from otteroad.models.scenario_events.projects.ScenarioObjectsUpdated import (
    ScenarioObjectsUpdated,
)
from otteroad.models.scenario_events.projects.ScenarioZonesUpdated import (
    ScenarioZonesUpdated,
)

from app.common.caching.caching_service import FileCache


@dataclass(frozen=True)
class CacheInvalidationRule:
    """
    Cache invalidation rule.

    method: cache method name (e.g. "social_economical_metrics")
    owner_id_getter: extracts owner_id from event (e.g. project_id or scenario_id)
    """
    method: str
    owner_id_getter: Callable[[Any], int]


class CacheInvalidationService:
    """Applies cache invalidation rules using FileCache."""

    def __init__(self, cache: FileCache) -> None:
        self._cache = cache

    def invalidate(self, event: Any, rules: Iterable[CacheInvalidationRule]) -> int:
        """
        Invalidate cache entries for event according to rules.

        Returns:
            Total number of deleted files.
        """
        total_deleted = 0
        for rule in rules:
            owner_id = int(rule.owner_id_getter(event))
            deleted = self._cache.delete_all(rule.method, owner_id)
            total_deleted += deleted

            logger.info(
                f"Cache invalidation applied: method={rule.method} owner_id={owner_id} deleted_files={deleted}"
            )

        return total_deleted


class CacheInvalidationHandlerCore:
    """Shared handler logic without BaseMessageHandler inheritance (otteroad-friendly)."""

    def __init__(
        self,
        invalidation_service: CacheInvalidationService,
        producer: KafkaProducerClient,
        rules: list[CacheInvalidationRule],
    ) -> None:
        self._invalidation_service = invalidation_service
        self._producer = producer
        self._rules = rules

    async def process(self, event: Any) -> Any:
        logger.info(f"Received event: type={type(event)}")
        logger.info(
            f"Invalidate cache for project_id={getattr(event, 'project_id', None)} "
            f"scenario_id={getattr(event, 'scenario_id', None)}"
        )

        total_deleted = self._invalidation_service.invalidate(event, self._rules)

        logger.info(f"Cache invalidation completed: deleted_files={total_deleted}")
        return None


class ScenarioObjectsUpdatedHandler(BaseMessageHandler[ScenarioObjectsUpdated]):
    """Invalidates cache when ScenarioObjectsUpdated is received."""

    def __init__(self, cache: FileCache, producer: KafkaProducerClient) -> None:
        self._core = CacheInvalidationHandlerCore(
            invalidation_service=CacheInvalidationService(cache),
            producer=producer,
            rules=[
                CacheInvalidationRule(
                    method="social_economical_metrics",
                    owner_id_getter=lambda e: e.project_id,
                ),
            ],
        )
        super().__init__()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def handle(self, event: EventT, ctx: Message = None):
        return await self._core.process(event)


class ScenarioZonesUpdatedHandler(BaseMessageHandler[ScenarioZonesUpdated]):
    """Invalidates cache when ScenarioZonesUpdated is received."""

    def __init__(self, cache: FileCache, producer: KafkaProducerClient) -> None:
        self._core = CacheInvalidationHandlerCore(
            invalidation_service=CacheInvalidationService(cache),
            producer=producer,
            rules=[
                CacheInvalidationRule(
                    method="social_economical_metrics",
                    owner_id_getter=lambda e: e.project_id,
                ),
            ],
        )
        super().__init__()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def handle(self, event: EventT, ctx: Message = None):
        return await self._core.process(event)
