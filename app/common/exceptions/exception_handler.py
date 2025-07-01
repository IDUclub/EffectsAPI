import functools

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

from .http_exception_wrapper import http_exception

base_types = (int, float, str, bool, list, tuple, set, dict, None)


def async_ultimate_exception_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(e)
            raise http_exception(
                500,
                msg="Internal server error",
                _input={
                    "func": func.__name__,
                    "args": jsonable_encoder(
                        [i if type(i) in base_types else i.model_dump() for i in args]
                    ),
                    "kwargs": jsonable_encoder(
                        {
                            i: (v if type(v) in base_types else v.model_dump())
                            for i, v in kwargs.items()
                        }
                    ),
                },
                _detail={"error": repr(e)},
            ) from e

    return wrapper
