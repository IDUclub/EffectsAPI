from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.common.auth.keycloak_validator import (
    JWKSUnavailableError,
    TokenValidationError,
)
from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import token_validator

http_bearer = HTTPBearer(auto_error=False)

BearerCredentials = Annotated[HTTPAuthorizationCredentials | None, Depends(http_bearer)]


def _get_token_from_header(credentials: HTTPAuthorizationCredentials | None) -> str:
    if not credentials:
        raise http_exception(401, "Authorization header missing")

    token = credentials.credentials

    if not token:
        raise http_exception(401, "Token is missing in the authorization header")

    return token


async def verify_token(credentials: BearerCredentials) -> str:
    token = _get_token_from_header(credentials)

    try:
        await token_validator.validate(token)
    except TokenValidationError as exc:
        raise http_exception(401, "Invalid access token", _detail=str(exc))
    except JWKSUnavailableError as exc:
        raise http_exception(
            503, "Authorization server is unavailable", _detail=str(exc)
        )

    return token
