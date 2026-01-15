"""Task metrics reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram


@dataclass(frozen=True)
class TaskMetrics:
    """
    Aggregates Prometheus metrics related to tasks and provides
    convenient reporting methods.

    This class hides Prometheus primitives from business logic.
    """

    created_total: Counter
    cache_hit_total: Counter
    enqueued_total: Counter
    started_total: Counter
    finished_total: Counter
    duration_seconds: Histogram
    running: Gauge
    queue_size: Gauge

    def bind_queue_size(self, qsize_getter: Callable[[], int]) -> None:
        """
        Bind queue size metric to a callable that returns current size.

        Args:
            qsize_getter: Callable returning current queue size.
        """
        self.queue_size.set_function(qsize_getter)

    def on_created(self, method: str) -> None:
        self.created_total.labels(method=method).inc()

    def on_cache_hit(self, method: str) -> None:
        self.cache_hit_total.labels(method=method).inc()

    def on_enqueued(self, method: str) -> None:
        self.enqueued_total.labels(method=method).inc()

    def on_started(self, method: str) -> None:
        self.started_total.labels(method=method).inc()
        self.running.inc()

    def on_finished_success(self, method: str) -> None:
        self.finished_total.labels(method=method, status="success").inc()
        self.running.dec()

    def on_finished_failed(self, method: str) -> None:
        self.finished_total.labels(method=method, status="failed").inc()
        self.running.dec()

    def observe_duration(self, method: str, seconds: float) -> None:
        self.duration_seconds.labels(method=method).observe(seconds)
