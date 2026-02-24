# ============================================================================
# src/utils/logging.py
# ============================================================================
"""
Logging configuration and utilities for medical ingestion engine.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_json: bool = False
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        format_json: Whether to use JSON format
    """
    log_level = getattr(logging, level.upper())

    # Create formatters
    if format_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra'):
            log_data['extra'] = record.extra

        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding context to logs."""

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None

    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_performance(logger: logging.Logger, operation: str):
    """
    Decorator to log operation performance.

    Args:
        logger: Logger instance
        operation: Operation name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{operation} completed in {duration:.3f}s")
                return result

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{operation} failed after {duration:.3f}s: {str(e)}")
                raise

        return wrapper
    return decorator


class LogAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context to all log messages."""

    def process(self, msg, kwargs):
        """Add extra context to log message."""
        # Add context fields to extra
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        kwargs['extra'].update(self.extra)

        return msg, kwargs


def create_audit_logger(name: str, log_file: Path) -> logging.Logger:
    """
    Create dedicated audit logger.

    Args:
        name: Logger name
        log_file: Path to audit log file

    Returns:
        Audit logger instance
    """
    logger = logging.getLogger(f"audit.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create file handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)

    # Use JSON format for audit logs
    formatter = JsonFormatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
