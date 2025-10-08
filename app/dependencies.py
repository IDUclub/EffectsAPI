import sys
from pathlib import Path

from iduconfig import Config
from loguru import logger

from app.clients.urban_api_client import UrbanAPIClient
from app.common.api_handlers.json_api_handler import JSONAPIHandler
from app.common.caching.caching_service import FileCache
from app.common.utils.ids_convertation import EffectsUtils
from app.effects_api.effects_service import EffectsService
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

json_api_handler = JSONAPIHandler(config.get("URBAN_API"))
urban_api_client = UrbanAPIClient(json_api_handler)
file_cache = FileCache()
scenario_service = ScenarioService(urban_api_client)
effects_utils = EffectsUtils(urban_api_client)
effects_service = EffectsService(urban_api_client, file_cache, scenario_service)
