from prometheus_client import start_http_server

_server = None


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server."""
    global _server
    _server = start_http_server(port)


def stop_metrics_server() -> None:
    """Stop Prometheus metrics HTTP server."""
    global _server
    if _server is not None:
        _server.shutdown()
        _server = None
