# ============================================================================
# src/utils/__init__.py
# ============================================================================
"""
Utility modules for medical ingestion engine.
"""

from .exceptions import (
    MedicalIngestionError,
    DocumentProcessingError,
    PDFExtractionError,
    TableExtractionError,
    TextExtractionError,
    ValidationError,
    PlausibilityError,
    ReferenceRangeError,
    FHIRConversionError,
    FHIRValidationError,
    TemplateMatchError,
    TemplateNotFoundError,
    ClassificationError,
    ConfigurationError,
    ModelError,
    ModelLoadError,
    InferenceError,
    CacheError,
    SpecimenQualityError,
    CriticalValueError,
    ConflictResolutionError,
    UnitConversionError,
)

from .logging import (
    setup_logging,
    get_logger,
    LogContext,
    log_performance,
    create_audit_logger,
)

from .metrics import (
    MetricsCollector,
    Timer,
    PerformanceTracker,
    get_metrics,
    increment,
    set_gauge,
    record_value,
    record_time,
    time_operation,
)

from .file_utils import (
    ensure_directory,
    get_file_hash,
    get_file_size,
    is_pdf,
    list_pdfs,
    copy_file,
    move_file,
    delete_file,
    read_json,
    write_json,
    get_relative_path,
    sanitize_filename,
    get_unique_filename,
    clean_directory,
    get_directory_size,
)

__all__ = [
    # Exceptions
    'MedicalIngestionError',
    'DocumentProcessingError',
    'PDFExtractionError',
    'TableExtractionError',
    'TextExtractionError',
    'ValidationError',
    'PlausibilityError',
    'ReferenceRangeError',
    'FHIRConversionError',
    'FHIRValidationError',
    'TemplateMatchError',
    'TemplateNotFoundError',
    'ClassificationError',
    'ConfigurationError',
    'ModelError',
    'ModelLoadError',
    'InferenceError',
    'CacheError',
    'SpecimenQualityError',
    'CriticalValueError',
    'ConflictResolutionError',
    'UnitConversionError',
    # Logging
    'setup_logging',
    'get_logger',
    'LogContext',
    'log_performance',
    'create_audit_logger',
    # Metrics
    'MetricsCollector',
    'Timer',
    'PerformanceTracker',
    'get_metrics',
    'increment',
    'set_gauge',
    'record_value',
    'record_time',
    'time_operation',
    # File Utils
    'ensure_directory',
    'get_file_hash',
    'get_file_size',
    'is_pdf',
    'list_pdfs',
    'copy_file',
    'move_file',
    'delete_file',
    'read_json',
    'write_json',
    'get_relative_path',
    'sanitize_filename',
    'get_unique_filename',
    'clean_directory',
    'get_directory_size',
]
