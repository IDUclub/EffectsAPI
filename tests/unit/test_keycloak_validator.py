import time

import jwt
import pytest

from app.common.auth.keycloak_validator import (
    JWKSUnavailableError,
    KeycloakValidatorConfig,
    TokenValidationError,
)
from tests.conftest import ENCRYPTION_KID, SIGNING_KID


async def test_valid_token_is_accepted(build_validator, make_token):
    validator, _ = build_validator()

    claims = await validator.validate(make_token())

    assert claims["sub"] == "user-123"
    assert claims["preferred_username"] == "tester"


async def test_only_signing_keys_are_loaded(build_validator, make_token):
    validator, _ = build_validator()

    await validator.validate(make_token())

    assert SIGNING_KID in validator._keys
    assert ENCRYPTION_KID not in validator._keys


async def test_jwks_is_fetched_once_across_validations(build_validator, make_token):
    validator, calls = build_validator()

    await validator.validate(make_token())
    await validator.validate(make_token())

    assert calls["fetches"] == 1


async def test_token_signed_by_unknown_key_is_rejected(
    build_validator, make_token, foreign_key
):
    validator, _ = build_validator()

    with pytest.raises(TokenValidationError):
        await validator.validate(make_token(key=foreign_key))


async def test_expired_token_is_rejected(build_validator, make_token):
    validator, _ = build_validator()

    with pytest.raises(TokenValidationError, match="expired"):
        await validator.validate(make_token(exp=int(time.time()) - 3600))


async def test_token_from_another_issuer_is_rejected(build_validator, make_token):
    validator, _ = build_validator()

    with pytest.raises(TokenValidationError):
        await validator.validate(make_token(iss="https://evil.example/realms/IDU"))


async def test_unknown_key_id_is_rejected(build_validator, make_token):
    validator, _ = build_validator()

    with pytest.raises(TokenValidationError, match="unknown signing key"):
        await validator.validate(make_token(kid="rotated-away"))


async def test_unsigned_token_is_rejected(build_validator, validator_config):
    validator, _ = build_validator()
    unsigned = jwt.encode(
        {"iss": validator_config.issuer, "sub": "x", "iat": 1, "exp": 9999999999},
        key=None,
        algorithm="none",
        headers={"kid": SIGNING_KID},
    )

    with pytest.raises(TokenValidationError):
        await validator.validate(unsigned)


async def test_token_without_key_id_is_rejected(build_validator, signing_key):
    validator, _ = build_validator()
    no_kid = jwt.encode({"sub": "x"}, signing_key, algorithm="RS256")

    with pytest.raises(TokenValidationError, match="no 'kid'"):
        await validator.validate(no_kid)


async def test_malformed_token_is_rejected(build_validator):
    validator, _ = build_validator()

    with pytest.raises(TokenValidationError, match="malformed"):
        await validator.validate("not-a-jwt")


async def test_token_issued_slightly_ahead_of_our_clock_is_accepted(
    build_validator, make_token
):
    """Keycloak's clock runs marginally ahead of ours; without leeway this 401s."""
    validator, _ = build_validator()

    claims = await validator.validate(make_token(iat=int(time.time()) + 5))

    assert claims["sub"] == "user-123"


async def test_token_issued_far_in_the_future_is_rejected(build_validator, make_token):
    validator, _ = build_validator()

    with pytest.raises(TokenValidationError):
        await validator.validate(make_token(iat=int(time.time()) + 600))


async def test_token_expired_within_leeway_is_tolerated(build_validator, make_token):
    validator, _ = build_validator()

    claims = await validator.validate(make_token(exp=int(time.time()) - 5))

    assert claims["sub"] == "user-123"


async def test_audience_is_not_checked_by_default(build_validator, make_token):
    """Realm tokens are audienced at urban-api, never at effects."""
    validator, _ = build_validator()

    claims = await validator.validate(make_token(aud=["urban-api", "account"]))

    assert claims["aud"] == ["urban-api", "account"]


async def test_wrong_audience_is_rejected_when_verification_enabled(
    build_validator, make_token, validator_config
):
    strict = KeycloakValidatorConfig(
        auth_server_url=validator_config.auth_server_url,
        realm=validator_config.realm,
        audience="effects",
        verify_audience=True,
    )
    validator, _ = build_validator(config=strict)

    with pytest.raises(TokenValidationError):
        await validator.validate(make_token(aud="account"))


async def test_unreachable_realm_without_cached_keys_raises(
    build_validator, make_token
):
    validator, _ = build_validator(unavailable=True)

    with pytest.raises(JWKSUnavailableError):
        await validator.validate(make_token())


async def test_unreachable_realm_falls_back_to_cached_keys(build_validator, make_token):
    validator, _ = build_validator()
    await validator.validate(make_token())

    async def unavailable():
        raise JWKSUnavailableError("realm outage")

    validator._fetch_jwks = unavailable
    validator._fetched_at = 0.0

    claims = await validator.validate(make_token())

    assert claims["sub"] == "user-123"


async def test_unknown_key_id_does_not_refetch_jwks_on_every_call(
    build_validator, make_token
):
    """An attacker spraying random kids must not turn into load on Keycloak."""
    validator, calls = build_validator()
    await validator.validate(make_token())

    for _ in range(5):
        with pytest.raises(TokenValidationError):
            await validator.validate(make_token(kid="bogus"))

    assert calls["fetches"] == 1
