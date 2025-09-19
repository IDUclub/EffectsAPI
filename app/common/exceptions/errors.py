class DomainError(Exception):
    pass


class UpstreamApiError(DomainError):
    def __init__(self, msg, *, status=None, payload=None):
        super().__init__(msg)
        self.status = status
        self.payload = payload or {}


class NoFeaturesError(DomainError):
    pass
