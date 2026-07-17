import sys
from pathlib import Path

from idu_service_auth import KeycloakTokenConfig
from iduconfig import Config
from loguru import logger

from app.broker_handlers.cache_invalidation import CacheInvalidationService
from app.broker_handlers.scenario_updated_handler import (
    ScenarioObjectsUpdatedHandler,
    ScenarioZonesUpdatedHandler,
)
from app.clients.urban_api_client import UrbanAPIClient
from app.common.api_handlers.json_api_handler import JSONAPIHandler
from app.common.auth.keycloak_validator import (
    KeycloakTokenValidator,
    KeycloakValidatorConfig,
)
from app.common.auth.service_token import ServiceTokenProvider
from app.common.caching.caching_service import FileCache
from app.common.consumer_wrapper import ConsumerWrapper
from app.common.producer_wrapper import ProducerWrapper
from app.common.utils.effects_utils import EffectsUtils
from app.effects_api.effects_service import EffectsService
from app.effects_api.modules.context_service import ContextService
from app.effects_api.modules.scenario_service import ScenarioService

absolute_app_path = Path().absolute()
config = Config()

logger.remove()
log_level = "INFO"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <b>{message}</b>"
logger.add(sys.stderr, format=log_format, level=log_level, colorize=True)
logger.add(
    absolute_app_path / f"{config.get('LOG_NAME')}",
    format=log_format,
    level="INFO",
)

keycloak_url = config.get("KEYCLOAK_URL")
keycloak_realm = config.get("KEYCLOAK_REALM")
keycloak_client_id = config.get("KEYCLOAK_CLIENT_ID")

token_validator = KeycloakTokenValidator(
    KeycloakValidatorConfig(
        auth_server_url=keycloak_url,
        realm=keycloak_realm,
        audience=keycloak_client_id,
        verify_audience=False,
    )
)

service_token_provider = ServiceTokenProvider(
    KeycloakTokenConfig(
        auth_server_url=keycloak_url,
        realm=keycloak_realm,
        client_id=keycloak_client_id,
        client_secret=config.get("KEYCLOAK_CLIENT_SECRET"),
    )
)

json_api_handler = JSONAPIHandler(config.get("URBAN_API"))
urban_api_client = UrbanAPIClient(json_api_handler, service_token_provider)
file_cache = FileCache()
scenario_service = ScenarioService(urban_api_client)
effects_utils = EffectsUtils(urban_api_client)
context_service = ContextService(urban_api_client, file_cache)
effects_service = EffectsService(
    urban_api_client, file_cache, scenario_service, context_service, effects_utils
)

consumer = ConsumerWrapper()
producer = ProducerWrapper()

cache_invalidator = CacheInvalidationService(file_cache)

consumer.register_handler(
    ScenarioObjectsUpdatedHandler(cache_invalidator, producer.producer_service)
)
consumer.register_handler(
    ScenarioZonesUpdatedHandler(cache_invalidator, producer.producer_service)
)
