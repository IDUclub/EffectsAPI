import sys

from loguru import logger
from iduconfig import Config

from app.common.api_handler import APIHandler

logger.remove()
logger.add(sys.stderr, level="INFO")
log_level = "INFO"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <b>{message}</b>"
logger.add(
    sys.stderr,
    format=log_format,
    level=log_level,
    colorize=True
)

config = Config()

logger.add(
    f"{config.get('LOGS_FILE')}.log",
    format=log_format,
    level="INFO",
)

urban_api_handler = APIHandler(config.get("URBAN_API"))
