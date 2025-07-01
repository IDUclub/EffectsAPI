"""Exception handling middleware is defined here."""

import itertools
import json
import traceback

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .http_exception_wrapper import http_exception


class ExceptionHandlerMiddleware(
    BaseHTTPMiddleware
):  # pylint: disable=too-few-public-methods
    """Handle exceptions, so they become http response code 500 - Internal Server Error.

    If debug is activated in app configuration, then stack trace is returned, otherwise only a generic error message.
    Message is sent to logger error stream anyway.
    """

    def __init__(self, app: FastAPI):
        """Passing debug as a list with single element is a hack to be able to change the value
        on the application startup.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            request_info = {
                "method": request.method,
                "url": str(request.url),
                "path_params": dict(request.path_params),
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
            }
            try:
                request_info["body"] = await request.json()
            except:
                try:
                    request_info["body"] = str(await request.body())
                except:
                    request_info["body"] = "Could not read request body"
            if isinstance(e, HTTPException):
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "message": (
                            e.detail.get("msg")
                            if isinstance(e.detail, dict)
                            else str(e.detail)
                        ),
                        "error_type": e.__class__.__name__,
                        "request": request_info,
                        "detail": (
                            e.detail.get("detail")
                            if isinstance(e.detail, dict)
                            else None
                        ),
                    },
                )
            return JSONResponse(
                status_code=500,
                content={
                    "message": "Internal server error",
                    "error_type": e.__class__.__name__,
                    "request": request_info,
                    "detail": str(e),
                    "traceback": traceback.format_exc().splitlines(),
                },
            )
