"""Prometheus metrics for Effects API."""

from prometheus_client import Counter, Histogram


CACHE_INVALIDATION_EVENTS_TOTAL = Counter(
    "effects_cache_invalidation_events_total",
    "Total number of cache invalidation events received",
)

CACHE_INVALIDATION_SUCCESS_TOTAL = Counter(
    "effects_cache_invalidation_success_total",
    "Total number of cache invalidation events successfully processed",
)

CACHE_INVALIDATION_ERROR_TOTAL = Counter(
    "effects_cache_invalidation_error_total",
    "Total number of cache invalidation events failed during processing",
)

CACHE_INVALIDATION_DURATION_SECONDS = Histogram(
    "effects_cache_invalidation_duration_seconds",
    "Duration of cache invalidation processing",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
)

EFFECTS_TERRITORY_TRANSFORMATION_TOTAL = Counter(
    "effects_territory_transformation_total",
    "Total number of territory_transformation calls",
)

EFFECTS_TERRITORY_TRANSFORMATION_ERROR_TOTAL = Counter(
    "effects_territory_transformation_error_total",
    "Total number of failed territory_transformation calls",
)

EFFECTS_TERRITORY_TRANSFORMATION_DURATION_SECONDS = Histogram(
    "effects_territory_transformation_duration_seconds",
    "Duration of territory_transformation execution",
    buckets=(1, 2, 5, 10, 30, 60, 120, 300),
)
