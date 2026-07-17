import asyncio
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import jwt
from jwt import PyJWKSet
from loguru import logger

from app.common.exceptions.errors import DomainError


class TokenValidationError(DomainError):
    """Raised when an access token cannot be verified against the realm keys."""


class JWKSUnavailableError(DomainError):
    """Raised when realm signing keys cannot be retrieved from Keycloak."""


@dataclass(frozen=True, slots=True)
class KeycloakValidatorConfig:
    auth_server_url: str
    realm: str
    algorithms: tuple[str, ...] = ("RS256",)
    audience: str | None = None
    verify_audience: bool = False
    # Keycloak and this service run on different clocks, so a freshly minted token
    # can carry an `iat` a fraction of a second ahead of us; without tolerance PyJWT
    # rejects it outright.
    leeway_seconds: float = 30.0
    request_timeout_seconds: float = 10.0
    jwks_cache_ttl_seconds: float = 3600.0
    jwks_min_refresh_interval_seconds: float = 30.0

    @property
    def issuer(self) -> str:
        return f"{self.auth_server_url.rstrip('/')}/realms/{self.realm}"

    @property
    def jwks_uri(self) -> str:
        return f"{self.issuer}/protocol/openid-connect/certs"


class KeycloakTokenValidator:
    """Verify user access tokens locally against the realm JWKS."""

    def __init__(
        self,
        config: KeycloakValidatorConfig,
        *,
        session: aiohttp.ClientSession | None = None,
    ):
        self._config = config
        self._session = session
        self._owns_session = session is None
        self._keys: dict[str, Any] = {}
        self._fetched_at: float = 0.0
        self._last_fetch_attempt: float = 0.0
        self._lock = asyncio.Lock()

    async def validate(self, token: str) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise TokenValidationError("malformed token header") from exc

        kid = header.get("kid")
        if not kid:
            raise TokenValidationError("token header has no 'kid'")

        key = await self._get_key(kid)

        try:
            return jwt.decode(
                token,
                key,
                algorithms=list(self._config.algorithms),
                issuer=self._config.issuer,
                leeway=self._config.leeway_seconds,
                audience=(
                    self._config.audience if self._config.verify_audience else None
                ),
                options={
                    "verify_aud": self._config.verify_audience,
                    "require": ["exp", "iat", "iss"],
                },
            )
        except jwt.ExpiredSignatureError as exc:
            raise TokenValidationError("token has expired") from exc
        except jwt.PyJWTError as exc:
            raise TokenValidationError("token verification failed") from exc

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_key(self, kid: str) -> Any:
        cached = self._keys.get(kid)
        if cached is not None and not self._is_stale():
            return cached

        try:
            await self._refresh_keys()
        except JWKSUnavailableError:
            if cached is None:
                raise
            logger.warning("JWKS refresh failed, verifying with cached signing key")
            return cached

        key = self._keys.get(kid)
        if key is None:
            raise TokenValidationError(f"unknown signing key id '{kid}'")
        return key

    def _is_stale(self) -> bool:
        return (
            time.monotonic() - self._fetched_at >= self._config.jwks_cache_ttl_seconds
        )

    async def _refresh_keys(self) -> None:
        async with self._lock:
            now = time.monotonic()
            throttled = (
                now - self._last_fetch_attempt
                < self._config.jwks_min_refresh_interval_seconds
            )
            if self._keys and throttled and not self._is_stale():
                return

            self._last_fetch_attempt = now
            raw = await self._fetch_jwks()

            try:
                key_set = PyJWKSet.from_dict(raw)
            except jwt.PyJWTError as exc:
                raise JWKSUnavailableError(
                    "realm returned no usable signing keys"
                ) from exc

            self._keys = {key.key_id: key.key for key in key_set.keys if key.key_id}
            self._fetched_at = now
            logger.info(f"Loaded {len(self._keys)} signing keys from realm JWKS")

    async def _fetch_jwks(self) -> dict[str, Any]:
        session = self._get_session()
        try:
            async with session.get(self._config.jwks_uri) as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise JWKSUnavailableError(
                f"failed to fetch JWKS from {self._config.jwks_uri}"
            ) from exc

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self._config.request_timeout_seconds
                )
            )
            self._owns_session = True
        return self._session
