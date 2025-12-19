from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, TypeVar

from confluent_kafka import Message
from loguru import logger
from otteroad import BaseMessageHandler, KafkaProducerClient
from otteroad.consumer.handlers.base import EventT

from app.common.caching.caching_service import FileCache

TEvent = TypeVar("TEvent")


@dataclass(frozen=True)
class CacheInvalidationRule:
    """
    Describes what cache to invalidate and how to extract owner_id from an event.

    method: cache method name (e.g. "social_economical_metrics").
    owner_id_getter: function that returns owner_id from event (e.g. project_id or scenario_id).
    """
    method: str
    owner_id_getter: Callable[[Any], int]


class CacheInvalidationService:
    """Applies cache invalidation rules using FileCache."""

    def __init__(self, cache: FileCache) -> None:
        self._cache = cache

    def invalidate(self, event: Any, rules: Iterable[CacheInvalidationRule]) -> int:
        """
        Invalidate cache for an event using given rules.

        Returns:
            Total number of deleted files.
        """
        total_deleted = 0
        for rule in rules:
            owner_id = int(rule.owner_id_getter(event))
            deleted = self._cache.delete_all(rule.method, owner_id)
            total_deleted += deleted

            logger.info(
                f"Cache invalidation rule applied: method={rule.method} owner_id={owner_id} deleted_files={deleted}"
            )

        return total_deleted


class BaseCacheInvalidationHandler(Generic[TEvent], BaseMessageHandler[TEvent]):
    """Base handler for cache invalidation with easy extension via rules."""

    def __init__(
        self,
        invalidation_service: CacheInvalidationService,
        producer: KafkaProducerClient,
        rules: list[CacheInvalidationRule],
    ) -> None:
        self._invalidation_service = invalidation_service
        self._producer = producer
        self._rules = rules
        super().__init__()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def handle(self, event: EventT, ctx: Message = None):
        logger.info(f"Received event: type={type(event)}")
        total_deleted = self._invalidation_service.invalidate(event, self._rules)
        logger.info(f"Cache invalidation completed: deleted_files={total_deleted}")
        return await self._producer.send(event)
