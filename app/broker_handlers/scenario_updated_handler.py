from __future__ import annotations

from confluent_kafka import Message
from otteroad import BaseMessageHandler, KafkaProducerClient
from otteroad.consumer.handlers.base import EventT
from otteroad.models.scenario_events.projects.ScenarioObjectsUpdated import ScenarioObjectsUpdated
from otteroad.models.scenario_events.projects.ScenarioZonesUpdated import ScenarioZonesUpdated

from app.broker_handlers.cache_invalidation import (
    CacheInvalidationMixin,
    CacheInvalidationRule,
    CacheInvalidationService,
)


_SOCIAL_RULES = [
    CacheInvalidationRule(method="social_economical_metrics", owner_id_getter=lambda e: e.project_id),
    CacheInvalidationRule(method="territory_transformation", owner_id_getter=lambda e: e.scenario_id)
]


class ScenarioObjectsUpdatedHandler(BaseMessageHandler[ScenarioObjectsUpdated], CacheInvalidationMixin):
    def __init__(self, invalidation_service: CacheInvalidationService, producer: KafkaProducerClient) -> None:
        CacheInvalidationMixin.__init__(self, invalidation_service, producer, _SOCIAL_RULES)
        BaseMessageHandler.__init__(self)

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def handle(self, event: EventT, ctx: Message = None):
        return await self._handle_cache_invalidation(event, ctx)


class ScenarioZonesUpdatedHandler(BaseMessageHandler[ScenarioZonesUpdated], CacheInvalidationMixin):
    def __init__(self, invalidation_service: CacheInvalidationService, producer: KafkaProducerClient) -> None:
        CacheInvalidationMixin.__init__(self, invalidation_service, producer, _SOCIAL_RULES)
        BaseMessageHandler.__init__(self)

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def handle(self, event: EventT, ctx: Message = None):
        return await self._handle_cache_invalidation(event, ctx)
