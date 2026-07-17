from idu_service_auth import KeycloakTokenClient, KeycloakTokenConfig


class ServiceTokenProvider:
    """Own the lifecycle of the Keycloak client_credentials token for outgoing calls."""

    def __init__(self, config: KeycloakTokenConfig):
        self.client = KeycloakTokenClient(config)

    async def start(self) -> None:
        await self.client.start_background_refresh()

    async def stop(self) -> None:
        await self.client.stop_background_refresh()
        await self.client.aclose()

    async def get_token(self) -> str:
        return await self.client.get_access_token()

    async def get_authorization_headers(self) -> dict[str, str]:
        return await self.client.get_authorization_headers()
