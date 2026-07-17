import os

import pytest

from app.common.auth.keycloak_validator import (
    KeycloakTokenValidator,
    KeycloakValidatorConfig,
    TokenValidationError,
)

idu_service_auth = pytest.importorskip("idu_service_auth")
KeycloakTokenClient = idu_service_auth.KeycloakTokenClient
KeycloakTokenConfig = idu_service_auth.KeycloakTokenConfig

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not all(
            (KEYCLOAK_URL, KEYCLOAK_REALM, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET)
        ),
        reason="KEYCLOAK_* environment variables are not configured",
    ),
]


@pytest.fixture
async def service_token_client():
    client = KeycloakTokenClient(
        KeycloakTokenConfig(
            auth_server_url=KEYCLOAK_URL,
            realm=KEYCLOAK_REALM,
            client_id=KEYCLOAK_CLIENT_ID,
            client_secret=KEYCLOAK_CLIENT_SECRET,
        )
    )
    yield client
    await client.aclose()


@pytest.fixture
async def live_validator():
    validator = KeycloakTokenValidator(
        KeycloakValidatorConfig(
            auth_server_url=KEYCLOAK_URL,
            realm=KEYCLOAK_REALM,
            audience=KEYCLOAK_CLIENT_ID,
        )
    )
    yield validator
    await validator.aclose()


async def test_service_token_is_issued(service_token_client):
    token = await service_token_client.get_access_token()

    assert token


async def test_authorization_headers_are_bearer(service_token_client):
    headers = await service_token_client.get_authorization_headers()

    assert headers["Authorization"].startswith("Bearer ")


async def test_service_token_is_cached_between_calls(service_token_client):
    first = await service_token_client.get_access_token()
    second = await service_token_client.get_access_token()

    assert first == second


async def test_realm_token_passes_our_validator(service_token_client, live_validator):
    token = await service_token_client.get_access_token()

    claims = await live_validator.validate(token)

    assert claims["iss"] == live_validator._config.issuer
    assert claims["azp"] == KEYCLOAK_CLIENT_ID


async def test_freshly_minted_token_survives_clock_skew(
    service_token_client, live_validator
):
    """A token minted moments ago carries an `iat` ahead of our clock."""
    token = await service_token_client.get_access_token(force_refresh=True)

    claims = await live_validator.validate(token)

    assert claims["azp"] == KEYCLOAK_CLIENT_ID


async def test_tampered_realm_token_is_rejected(service_token_client, live_validator):
    token = await service_token_client.get_access_token()
    header, payload, signature = token.split(".")
    flipped = "B" if signature[0] != "B" else "C"

    with pytest.raises(TokenValidationError):
        await live_validator.validate(f"{header}.{payload}.{flipped}{signature[1:]}")
