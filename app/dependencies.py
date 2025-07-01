import sys
from pathlib import Path

from iduconfig import Config
from loguru import logger

from app.gateways.urban_api_gateway import UrbanAPIGateway

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

urban_api_gateway = UrbanAPIGateway(config.get("URBAN_API"))
