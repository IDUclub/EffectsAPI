import time
from typing import Any, Callable

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from app.common.auth.keycloak_validator import (
    JWKSUnavailableError,
    KeycloakTokenValidator,
    KeycloakValidatorConfig,
)

SIGNING_KID = "sig-key"
ENCRYPTION_KID = "enc-key"


@pytest.fixture(scope="session")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def foreign_key() -> rsa.RSAPrivateKey:
    """A key the realm never published, standing in for a forged token."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def validator_config() -> KeycloakValidatorConfig:
    return KeycloakValidatorConfig(
        auth_server_url="https://keycloak.example.org",
        realm="IDU",
        audience="effects",
    )


@pytest.fixture(scope="session")
def jwks_document(signing_key: rsa.RSAPrivateKey) -> dict[str, Any]:
    """Mirror a realm JWKS, which also publishes a key PyJWT cannot verify with."""
    signing_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(
        signing_key.public_key(), as_dict=True
    )
    signing_jwk.update({"kid": SIGNING_KID, "use": "sig", "alg": "RS256"})
    encryption_jwk = dict(signing_jwk, kid=ENCRYPTION_KID, use="enc", alg="RSA-OAEP")
    return {"keys": [signing_jwk, encryption_jwk]}


@pytest.fixture
def make_token(
    validator_config: KeycloakValidatorConfig, signing_key: rsa.RSAPrivateKey
) -> Callable[..., str]:
    def _make(
        *,
        key: rsa.RSAPrivateKey | None = None,
        kid: str = SIGNING_KID,
        **overrides: Any,
    ) -> str:
        now = int(time.time())
        claims = {
            "iss": validator_config.issuer,
            "sub": "user-123",
            "aud": "account",
            "iat": now,
            "exp": now + 300,
            "preferred_username": "tester",
        }
        claims.update(overrides)
        return jwt.encode(
            claims, key or signing_key, algorithm="RS256", headers={"kid": kid}
        )

    return _make


@pytest.fixture
def build_validator(
    validator_config: KeycloakValidatorConfig, jwks_document: dict[str, Any]
) -> Callable[..., tuple[KeycloakTokenValidator, dict[str, int]]]:
    """Build a validator whose JWKS endpoint is replaced by a counting stub."""

    def _build(
        *,
        config: KeycloakValidatorConfig | None = None,
        unavailable: bool = False,
    ) -> tuple[KeycloakTokenValidator, dict[str, int]]:
        validator = KeycloakTokenValidator(config or validator_config)
        calls = {"fetches": 0}

        async def fake_fetch() -> dict[str, Any]:
            calls["fetches"] += 1
            if unavailable:
                raise JWKSUnavailableError("simulated realm outage")
            return jwks_document

        validator._fetch_jwks = fake_fetch
        return validator, calls

    return _build
