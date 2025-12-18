from typing import Protocol


from confluent_kafka import Message
from iduconfig import Config
from loguru import logger
from otteroad import BaseMessageHandler, KafkaProducerClient
from otteroad.consumer.handlers.base import EventT
from otteroad.models.scenario_events.projects.ScenarioObjectsUpdated import ScenarioObjectsUpdated
from otteroad.models.scenario_events.projects.ScenarioZonesUpdated import ScenarioZonesUpdated

from app.clients.urban_api_client import UrbanAPIClient
from app.effects_api.dto.socio_economic_project_dto import SocioEconomicByProjectDTO
from app.effects_api.effects_service import EffectsService

# ScenarioZonesUpdated
class ScenarioObjectsUpdatedHandler(BaseMessageHandler[ScenarioObjectsUpdated]):
    def __init__(
            self,
            effects: EffectsService,
            producer: KafkaProducerClient,
            urban_api: UrbanAPIClient,
            config: Config

    ):
        self.effects = effects
        self.producer = producer
        self.urban_api = urban_api
        self.config = config
        super().__init__()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def handle(self, event: EventT, ctx: Message = None):
        logger.info(f"Received {type(event)}")
        logger.info(
            f"scenario: {event.scenario_id}, project: {event.project_id}")
        scenario_id: int = event.scenario_id
        regional_scenario_response = await self.urban_api.get_scenario(scenario_id, self.config.get("URBAN_API_TOKEN"))
        regional_scenario_id = regional_scenario_response["parent_scenario"]["id"]
        params =  SocioEconomicByProjectDTO(
            project_id=event.project_id,
            regional_scenario_id=regional_scenario_id,
            force=True)
        await self.effects.evaluate_social_economical_metrics(self.config.get("URBAN_API_TOKEN"), params=params)
        logger.error(
            f"F22 was calculated for {event.project_id} project")
        return await self.producer.send(event)

