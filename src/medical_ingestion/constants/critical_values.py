# ============================================================================
# src/medical_ingestion/constants/critical_values.py
# ============================================================================
"""
Critical Value Thresholds
- Values that require immediate clinical attention
"""

CRITICAL_VALUES = {
    "hemoglobin": {"low": 7.0, "high": 20.0},
    "wbc": {"low": 1.0, "high": 30.0},
    "platelets": {"low": 20, "high": 1000},
    "glucose": {"low": 40, "high": 500},
    "sodium": {"low": 120, "high": 160},
    "potassium": {"low": 2.5, "high": 6.5},
    "creatinine": {"high": 5.0},
}
