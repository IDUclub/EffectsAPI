from prometheus_client import Counter, Histogram, Gauge


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
    buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600, 900, 1200),
)

EFFECTS_TASKS_CREATED_TOTAL = Counter(
    "effects_tasks_created_total",
    "Total number of tasks created",
    labelnames=("method",),
)

EFFECTS_TASKS_CACHE_HIT_TOTAL = Counter(
    "effects_tasks_cache_hit_total",
    "Total number of tasks served from cache (no execution needed)",
    labelnames=("method",),
)

EFFECTS_TASKS_ENQUEUED_TOTAL = Counter(
    "effects_tasks_enqueued_total",
    "Total number of tasks enqueued for execution",
    labelnames=("method",),
)

EFFECTS_TASKS_STARTED_TOTAL = Counter(
    "effects_tasks_started_total",
    "Total number of tasks started execution",
    labelnames=("method",),
)

EFFECTS_TASKS_DONE_TOTAL = Counter(
    "effects_tasks_done_total",
    "Total number of tasks finished successfully",
    labelnames=("method",),
)

EFFECTS_TASKS_FAILED_TOTAL = Counter(
    "effects_tasks_failed_total",
    "Total number of tasks failed during execution",
    labelnames=("method",),
)

EFFECTS_TASK_DURATION_SECONDS = Histogram(
    "effects_task_duration_seconds",
    "Task execution duration in seconds",
    labelnames=("method",),
    buckets=(0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)

EFFECTS_TASKS_QUEUE_SIZE = Gauge(
    "effects_tasks_queue_size",
    "Current number of tasks waiting in queue",
)

EFFECTS_TASKS_RUNNING = Gauge(
    "effects_tasks_running",
    "Current number of tasks running",
)

# --- Service entrypoints metrics ---

EFFECTS_VALUES_TRANSFORMATION_TOTAL = Counter(
    "effects_values_transformation_total",
    "Total number of values_transformation calls",
)

EFFECTS_VALUES_TRANSFORMATION_ERROR_TOTAL = Counter(
    "effects_values_transformation_error_total",
    "Total number of failed values_transformation calls",
)

EFFECTS_VALUES_TRANSFORMATION_DURATION_SECONDS = Histogram(
    "effects_values_transformation_duration_seconds",
    "Duration of values_transformation execution",
    buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600),
)

EFFECTS_VALUES_ORIENTED_REQUIREMENTS_TOTAL = Counter(
    "effects_values_oriented_requirements_total",
    "Total number of values_oriented_requirements calls",
)

EFFECTS_VALUES_ORIENTED_REQUIREMENTS_ERROR_TOTAL = Counter(
    "effects_values_oriented_requirements_error_total",
    "Total number of failed values_oriented_requirements calls",
)

EFFECTS_VALUES_ORIENTED_REQUIREMENTS_DURATION_SECONDS = Histogram(
    "effects_values_oriented_requirements_duration_seconds",
    "Duration of values_oriented_requirements execution",
    buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600),
)

EFFECTS_SOCIO_ECONOMICAL_METRICS_TOTAL = Counter(
    "effects_social_economical_metrics_total",
    "Total number of evaluate_social_economical_metrics calls",
)

EFFECTS_SOCIO_ECONOMICAL_METRICS_ERROR_TOTAL = Counter(
    "effects_social_economical_metrics_error_total",
    "Total number of failed evaluate_social_economical_metrics calls",
)

EFFECTS_SOCIO_ECONOMICAL_METRICS_DURATION_SECONDS = Histogram(
    "effects_social_economical_metrics_duration_seconds",
    "Duration of evaluate_social_economical_metrics execution",
    buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600),
)
