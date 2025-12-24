# app/metrics_server.py
from prometheus_client import start_http_server

_server = None


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server."""
    global _server
    _server = start_http_server(port)
