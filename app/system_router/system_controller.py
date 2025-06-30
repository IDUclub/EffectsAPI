from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import absolute_app_path, config

LOGS_PATH = absolute_app_path / f"{config.get('LOG_NAME')}"
system_router = APIRouter(prefix="/system", tags=["System"])


@system_router.get("/logs")
async def get_logs():
    """
    Get logs file from app
    """

    try:
        return FileResponse(
            LOGS_PATH,
            media_type="application/octet-stream",
            filename=f"effects.log",
        )
    except FileNotFoundError as e:
        raise http_exception(
            status_code=404,
            msg="Log file not found",
            _input={"log_path": LOGS_PATH, "log_file_name": config.get("LOG_NAME")},
            _detail={"error": repr(e)},
        ) from e
    except Exception as e:
        raise http_exception(
            status_code=500,
            msg="Internal server error during reading logs",
            _input={"log_path": LOGS_PATH, "log_file_name": config.get("LOG_NAME")},
            _detail={"error": repr(e)},
        ) from e
