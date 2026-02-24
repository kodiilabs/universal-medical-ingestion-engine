# ============================================================================
# src/utils/metrics.py
# ============================================================================
"""
Performance metrics tracking for medical ingestion engine.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import statistics


@dataclass
class MetricPoint:
    """Single metric measurement."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimerResult:
    """Result of a timed operation."""
    duration: float
    start_time: datetime
    end_time: datetime
    operation: str


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1) -> None:
        """
        Increment counter.

        Args:
            name: Counter name
            value: Increment amount
        """
        self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """
        Set gauge value.

        Args:
            name: Gauge name
            value: Current value
        """
        self._gauges[name] = value

    def record_value(self, name: str, value: float) -> None:
        """
        Record value in histogram.

        Args:
            name: Histogram name
            value: Value to record
        """
        self._histograms[name].append(value)

    def record_time(self, name: str, duration: float) -> None:
        """
        Record operation duration.

        Args:
            name: Operation name
            duration: Duration in seconds
        """
        self._timers[name].append(duration)

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        return self._gauges.get(name)

    def get_histogram_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get histogram statistics.

        Returns:
            Dict with min, max, mean, median, p95, p99
        """
        values = self._histograms.get(name, [])
        if not values:
            return None

        sorted_values = sorted(values)
        count = len(values)

        return {
            'count': count,
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': sorted_values[int(count * 0.95)] if count > 0 else 0,
            'p99': sorted_values[int(count * 0.99)] if count > 0 else 0,
        }

    def get_timer_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get timer statistics."""
        return self.get_histogram_stats(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {
                name: self.get_histogram_stats(name)
                for name in self._histograms.keys()
            },
            'timers': {
                name: self.get_timer_stats(name)
                for name in self._timers.keys()
            }
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timers.clear()


class Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, operation: str):
        self.collector = collector
        self.operation = operation
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        self.collector.record_time(self.operation, self.duration)


class PerformanceTracker:
    """Track performance of document processing."""

    def __init__(self):
        self.metrics = MetricsCollector()
        self._document_times: Dict[str, float] = {}
        self._stage_times: Dict[str, Dict[str, float]] = defaultdict(dict)

    def start_document(self, document_id: str) -> None:
        """Start tracking document processing."""
        self._document_times[document_id] = time.time()

    def end_document(self, document_id: str) -> float:
        """
        End tracking document processing.

        Returns:
            Total processing time
        """
        if document_id not in self._document_times:
            return 0.0

        duration = time.time() - self._document_times[document_id]
        self.metrics.record_time('document_processing', duration)
        self.metrics.increment('documents_processed')

        del self._document_times[document_id]
        return duration

    def record_stage(self, document_id: str, stage: str, duration: float) -> None:
        """Record stage duration for document."""
        self._stage_times[document_id][stage] = duration
        self.metrics.record_time(f'stage_{stage}', duration)

    def time_operation(self, operation: str) -> Timer:
        """
        Create timer for operation.

        Args:
            operation: Operation name

        Returns:
            Timer context manager
        """
        return Timer(self.metrics, operation)

    def record_extraction(self, field_count: int, confidence: float) -> None:
        """Record extraction results."""
        self.metrics.increment('fields_extracted', field_count)
        self.metrics.record_value('extraction_confidence', confidence)

    def record_validation(self, passed: bool) -> None:
        """Record validation result."""
        if passed:
            self.metrics.increment('validations_passed')
        else:
            self.metrics.increment('validations_failed')

    def record_error(self, error_type: str) -> None:
        """Record error occurrence."""
        self.metrics.increment(f'error_{error_type}')

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        all_metrics = self.metrics.get_all_metrics()

        # Calculate success rate
        total_docs = all_metrics['counters'].get('documents_processed', 0)
        errors = sum(
            count for name, count in all_metrics['counters'].items()
            if name.startswith('error_')
        )

        success_rate = ((total_docs - errors) / total_docs * 100) if total_docs > 0 else 0

        return {
            'total_documents': total_docs,
            'success_rate': success_rate,
            'errors': errors,
            'metrics': all_metrics
        }


# Global metrics instance
_global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _global_metrics


def increment(name: str, value: int = 1) -> None:
    """Increment global counter."""
    _global_metrics.increment(name, value)


def set_gauge(name: str, value: float) -> None:
    """Set global gauge."""
    _global_metrics.set_gauge(name, value)


def record_value(name: str, value: float) -> None:
    """Record value in global histogram."""
    _global_metrics.record_value(name, value)


def record_time(name: str, duration: float) -> None:
    """Record duration in global timer."""
    _global_metrics.record_time(name, duration)


def time_operation(operation: str) -> Timer:
    """Create timer for operation using global metrics."""
    return Timer(_global_metrics, operation)
