from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from confluent_kafka import Message
from loguru import logger
from otteroad import KafkaProducerClient

from app.common.caching.caching_service import FileCache


@dataclass(frozen=True)
class CacheInvalidationRule:
    """
    Cache invalidation rule.

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


class CacheInvalidationMixin:
    """
    Shared handler logic for cache invalidation.
    """
    def __init__(
        self,
        invalidation_service: CacheInvalidationService,
        producer: KafkaProducerClient,
        rules: list[CacheInvalidationRule],
    ) -> None:
        self._invalidation_service = invalidation_service
        self._producer = producer
        self._rules = rules

    async def _handle_cache_invalidation(self, event: Any, ctx: Message | None = None) -> None:
        logger.info(f"Received event: type={type(event)}")
        logger.info(
            f"Invalidate cache for project_id={getattr(event, 'project_id', None)} "
            f"scenario_id={getattr(event, 'scenario_id', None)}"
        )

        total_deleted = await asyncio.to_thread(
            self._invalidation_service.invalidate,
            event,
            self._rules,
        )
        logger.info(f"Cache invalidation completed: deleted_files={total_deleted}")

        return None
