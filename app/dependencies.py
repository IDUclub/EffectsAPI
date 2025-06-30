import sys
from pathlib import Path

from loguru import logger
from iduconfig import Config

from app.gateways.urban_api_gateway import UrbanAPIGateway

logger.remove()
log_level = "INFO"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <b>{message}</b>"
logger.add(
    sys.stderr,
    format=log_format,
    level=log_level,
    colorize=True
)
logger.add(".log", level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)

absolute_app_path = Path().absolute()
config = Config()

logger.add(
    absolute_app_path / f"{config.get('LOG_NAME')}",
    format=log_format,
    level="INFO",
)

urban_api_gateway = UrbanAPIGateway(config.get("URBAN_API"))
