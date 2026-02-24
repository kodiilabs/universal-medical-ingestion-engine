# ============================================================================
# src/utils/exceptions.py
# ============================================================================
"""
Custom exceptions for medical ingestion engine.
"""


class MedicalIngestionError(Exception):
    """Base exception for all medical ingestion errors."""
    pass


class DocumentProcessingError(MedicalIngestionError):
    """Error during document processing."""
    pass


class PDFExtractionError(DocumentProcessingError):
    """Error extracting data from PDF."""
    pass


class TableExtractionError(PDFExtractionError):
    """Error extracting tables from PDF."""
    pass


class TextExtractionError(PDFExtractionError):
    """Error extracting text from PDF."""
    pass


class ValidationError(MedicalIngestionError):
    """Error during data validation."""
    pass


class PlausibilityError(ValidationError):
    """Value fails plausibility check."""
    pass


class ReferenceRangeError(ValidationError):
    """Invalid reference range."""
    pass


class FHIRConversionError(MedicalIngestionError):
    """Error converting to FHIR format."""
    pass


class FHIRValidationError(FHIRConversionError):
    """FHIR resource validation failed."""
    pass


class TemplateMatchError(MedicalIngestionError):
    """Error matching document to template."""
    pass


class TemplateNotFoundError(TemplateMatchError):
    """No matching template found."""
    pass


class ClassificationError(MedicalIngestionError):
    """Error classifying document type."""
    pass


class ConfigurationError(MedicalIngestionError):
    """Invalid configuration."""
    pass


class ModelError(MedicalIngestionError):
    """Error with ML model."""
    pass


class ModelLoadError(ModelError):
    """Error loading ML model."""
    pass


class InferenceError(ModelError):
    """Error during model inference."""
    pass


class CacheError(MedicalIngestionError):
    """Error with caching system."""
    pass


class FileNotFoundError(MedicalIngestionError):
    """Required file not found."""
    pass


class InvalidFileFormatError(MedicalIngestionError):
    """Invalid file format."""
    pass


class SpecimenQualityError(MedicalIngestionError):
    """Specimen quality issue detected."""
    def __init__(self, message: str, severity: str = "moderate"):
        super().__init__(message)
        self.severity = severity


class CriticalValueError(MedicalIngestionError):
    """Critical lab value detected."""
    def __init__(self, message: str, field_name: str, value: float):
        super().__init__(message)
        self.field_name = field_name
        self.value = value


class ConflictResolutionError(ValidationError):
    """Cannot resolve validation conflict."""
    def __init__(self, message: str, field_name: str, rule_valid: bool, ai_valid: bool):
        super().__init__(message)
        self.field_name = field_name
        self.rule_valid = rule_valid
        self.ai_valid = ai_valid


class UnitConversionError(MedicalIngestionError):
    """Error converting units."""
    def __init__(self, message: str, from_unit: str, to_unit: str):
        super().__init__(message)
        self.from_unit = from_unit
        self.to_unit = to_unit
