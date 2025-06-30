from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException
from loguru import logger

from .http_exception_wrapper import http_exception

base_types = (int, float, str, bool, None)


async def handle_controller_exception(func, is_async: bool = True, **kwargs):
    try:
        if is_async:
            return await func(**kwargs)
        else:
            return func(**kwargs)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise http_exception(
            500,
            msg="Internal server error",
            _input={
                "func": str(func),
                "is_async": is_async,
                "kwargs": jsonable_encoder(
                    {i: (v if type(v) in base_types else v.as_dict()) for i, v in kwargs.items()}),
            },
            _detail={"error": repr(e)},
        ) from e
