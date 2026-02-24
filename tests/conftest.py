# ============================================================================
# FILE: tests/conftest.py
# ============================================================================
"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import sys


from src.medical_ingestion.core.context.extracted_value import ExtractedValue
from src.medical_ingestion.core.context.processing_context import ProcessingContext



@pytest.fixture
def sample_lab_text():
    """Sample lab report text for testing"""
    return """
    Quest Diagnostics Laboratory Report
    
    Patient: John Doe
    Date: 2024-01-15
    
    COMPLETE BLOOD COUNT (CBC)
    
    Test                Result      Reference Range    Flag
    ----------------------------------------------------------------
    WBC                 7.2         4.5-11.0 K/uL
    RBC                 4.8         4.5-5.5 M/uL
    Hemoglobin          14.2        13.5-17.5 g/dL
    Hematocrit          42.1        38.8-50.0 %
    Platelets           245         150-400 K/uL
    """


@pytest.fixture
def sample_radiology_text():
    """Sample radiology report text"""
    return """
    RADIOLOGY REPORT
    
    Examination: Chest X-Ray PA and Lateral
    
    CLINICAL INDICATION: Cough
    
    COMPARISON: None available
    
    FINDINGS:
    The lungs are clear without focal consolidation, effusion, or pneumothorax.
    The cardiac silhouette is normal in size and contour.
    
    IMPRESSION:
    Normal chest radiograph.
    """


@pytest.fixture
def sample_context():
    """Create sample processing context"""
    context = ProcessingContext(
        document_path=Path("test_lab.pdf"),
        patient_demographics={"age": 45, "sex": "M"}
    )
    context.document_type = "lab"
    context.raw_text = "Sample lab report text"
    return context


@pytest.fixture
def sample_extracted_values():
    """Create sample extracted lab values"""
    return [
        ExtractedValue(
            field_name="hemoglobin",
            value=14.2,
            unit="g/dL",
            confidence=0.95,
            extraction_method="template",
            reference_min=13.5,
            reference_max=17.5,
            abnormal_flag=None
        ),
        ExtractedValue(
            field_name="wbc",
            value=7.2,
            unit="K/uL",
            confidence=0.98,
            extraction_method="template",
            reference_min=4.5,
            reference_max=11.0,
            abnormal_flag=None
        ),
        ExtractedValue(
            field_name="potassium",
            value=6.8,
            unit="mmol/L",
            confidence=0.92,
            extraction_method="template",
            reference_min=3.5,
            reference_max=5.0,
            abnormal_flag="H"
        )
    ]


@pytest.fixture
def sample_patient_history():
    """Create sample patient history for temporal analysis"""
    base_date = datetime.now() - timedelta(days=90)
    
    return [
        {
            "date": base_date,
            "values": {
                "hemoglobin": 12.5,
                "wbc": 7.5,
                "platelets": 250
            }
        },
        {
            "date": base_date + timedelta(days=30),
            "values": {
                "hemoglobin": 11.2,
                "wbc": 7.3,
                "platelets": 240
            }
        },
        {
            "date": base_date + timedelta(days=60),
            "values": {
                "hemoglobin": 9.8,
                "wbc": 7.1,
                "platelets": 235
            }
        }
    ]


@pytest.fixture
def temp_pdf_path(tmp_path):
    """Create temporary PDF path for testing"""
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.touch()
    return pdf_path


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "model_path": Path("models/medgemma"),
        "cache_dir": Path("cache/test"),
        "use_gpu": False
    }


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings after each test"""
    yield
    # Settings reset happens automatically via pydantic


@pytest.fixture
def sample_template():
    """Sample lab template for testing"""
    return {
        "id": "test_cbc_v1",
        "vendor": "Test Lab",
        "test_type": "cbc",
        "header_pattern": "Test Lab.*CBC",
        "vendor_markers": ["Test Lab", "test.com"],
        "required_fields": ["WBC", "Hemoglobin", "Platelets"],
        "field_mappings": {
            "wbc": {
                "pdf_name": "WBC",
                "loinc_code": "6690-2",
                "unit": "K/uL",
                "value_column": 1,
                "ref_range_column": 2
            },
            "hemoglobin": {
                "pdf_name": "Hemoglobin",
                "loinc_code": "718-7",
                "unit": "g/dL",
                "value_column": 1,
                "ref_range_column": 2
            }
        }
    }