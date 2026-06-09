"""Prometheus metric factory helpers shared across projects."""

from prometheus_client import Counter, Histogram


def make_counter(name: str, description: str, labels: list[str] | None = None) -> Counter:
    return Counter(name, description, labels or [])


def make_histogram(
    name: str,
    description: str,
    labels: list[str] | None = None,
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
) -> Histogram:
    return Histogram(name, description, labels or [], buckets=buckets)
