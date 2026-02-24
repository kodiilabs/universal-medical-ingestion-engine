# ============================================================================
# FILE: tests/unit/test_plausibility.py
# ============================================================================
"""
Unit tests for plausibility checker
"""

import pytest
from src.validators.plausibility import (
    PlausibilityChecker,
    check_plausibility,
    get_plausibility_range
)


def test_plausibility_checker_init():
    """Test plausibility checker initialization"""
    checker = PlausibilityChecker()
    assert checker.ranges is not None
    assert "hemoglobin" in checker.ranges


def test_check_valid_hemoglobin():
    """Test valid hemoglobin value"""
    checker = PlausibilityChecker()
    is_plausible, reason = checker.check("hemoglobin", 14.2, "g/dL")

    assert is_plausible is True
    assert reason is None


def test_check_implausible_high_hemoglobin():
    """Test implausibly high hemoglobin (decimal error)"""
    checker = PlausibilityChecker()
    is_plausible, reason = checker.check("hemoglobin", 142.0, "g/dL")

    assert is_plausible is False
    assert "above plausible maximum" in reason


def test_check_implausible_low_hemoglobin():
    """Test implausibly low hemoglobin"""
    checker = PlausibilityChecker()
    is_plausible, reason = checker.check("hemoglobin", 1.0, "g/dL")

    assert is_plausible is False
    assert "below plausible minimum" in reason


def test_check_valid_potassium():
    """Test valid potassium value"""
    checker = PlausibilityChecker()
    is_plausible, reason = checker.check("potassium", 4.2, "mmol/L")

    assert is_plausible is True
    assert reason is None


def test_check_implausible_potassium():
    """Test implausibly high potassium"""
    checker = PlausibilityChecker()
    is_plausible, reason = checker.check("potassium", 42.0, "mmol/L")

    assert is_plausible is False
    assert "above plausible maximum" in reason


def test_check_unknown_test():
    """Test unknown test (should pass)"""
    checker = PlausibilityChecker()
    is_plausible, reason = checker.check("unknown_test", 999.9, "units")

    assert is_plausible is True
    assert reason is None


def test_check_wrong_unit():
    """Test value with wrong unit"""
    checker = PlausibilityChecker()
    is_plausible, reason = checker.check("hemoglobin", 14.2, "mg/dL")

    assert is_plausible is False
    assert "Unit mismatch" in reason


def test_suggest_correction_decimal_shift_right():
    """Test decimal correction suggestion (142 -> 14.2)"""
    checker = PlausibilityChecker()
    corrected = checker.suggest_correction("hemoglobin", 142.0, "g/dL")

    assert corrected == 14.2


def test_suggest_correction_double_shift():
    """Test double decimal shift (1420 -> 14.2)"""
    checker = PlausibilityChecker()
    corrected = checker.suggest_correction("hemoglobin", 1420.0, "g/dL")

    assert corrected == 14.2


def test_suggest_correction_decimal_shift_left():
    """Test decimal shift left (1.42 -> 14.2)"""
    checker = PlausibilityChecker()
    corrected = checker.suggest_correction("hemoglobin", 1.42, "g/dL")

    assert corrected == 14.2


def test_suggest_correction_no_correction_needed():
    """Test no correction needed for valid value"""
    checker = PlausibilityChecker()
    corrected = checker.suggest_correction("hemoglobin", 14.2, "g/dL")

    assert corrected is None


def test_suggest_correction_no_valid_correction():
    """Test no valid correction found"""
    checker = PlausibilityChecker()
    # Use a value that even after division won't be in range
    corrected = checker.suggest_correction("hemoglobin", 99999.0, "g/dL")

    assert corrected is None


def test_batch_check():
    """Test batch checking multiple values"""
    checker = PlausibilityChecker()
    values = {
        "hemoglobin": (14.2, "g/dL"),
        "potassium": (4.2, "mmol/L"),
        "glucose": (95.0, "mg/dL")
    }

    results = checker.batch_check(values)

    assert len(results) == 3
    assert results["hemoglobin"][0] is True
    assert results["potassium"][0] is True
    assert results["glucose"][0] is True


def test_batch_check_with_errors():
    """Test batch checking with some errors"""
    checker = PlausibilityChecker()
    values = {
        "hemoglobin": (142.0, "g/dL"),  # Too high
        "potassium": (4.2, "mmol/L"),   # Valid
        "glucose": (2000.0, "mg/dL")    # Too high
    }

    results = checker.batch_check(values)

    assert results["hemoglobin"][0] is False
    assert results["potassium"][0] is True
    assert results["glucose"][0] is False


def test_get_range():
    """Test getting plausibility range"""
    checker = PlausibilityChecker()
    range_info = checker.get_range("hemoglobin")

    assert range_info is not None
    assert len(range_info) == 3
    assert range_info[2] == "g/dL"


def test_get_range_unknown_test():
    """Test getting range for unknown test"""
    checker = PlausibilityChecker()
    range_info = checker.get_range("unknown_test")

    assert range_info is None


def test_convenience_function_check_plausibility():
    """Test convenience function"""
    result = check_plausibility("hemoglobin", 14.2, "g/dL")

    assert result is True


def test_convenience_function_get_range():
    """Test convenience function for getting range"""
    range_info = get_plausibility_range("hemoglobin")

    assert range_info is not None
    assert range_info[2] == "g/dL"


def test_glucose_plausibility():
    """Test glucose plausibility checks"""
    checker = PlausibilityChecker()

    # Valid glucose values
    assert checker.check("glucose", 70, "mg/dL")[0] is True
    assert checker.check("glucose", 200, "mg/dL")[0] is True

    # Invalid glucose values
    assert checker.check("glucose", 10, "mg/dL")[0] is False
    assert checker.check("glucose", 1500, "mg/dL")[0] is False


def test_sodium_plausibility():
    """Test sodium plausibility checks"""
    checker = PlausibilityChecker()

    # Valid sodium values
    assert checker.check("sodium", 140, "mmol/L")[0] is True
    assert checker.check("sodium", 120, "mmol/L")[0] is True

    # Invalid sodium values
    assert checker.check("sodium", 50, "mmol/L")[0] is False
    assert checker.check("sodium", 250, "mmol/L")[0] is False


def test_creatinine_plausibility():
    """Test creatinine plausibility checks"""
    checker = PlausibilityChecker()

    # Valid creatinine values
    assert checker.check("creatinine", 1.0, "mg/dL")[0] is True
    assert checker.check("creatinine", 5.0, "mg/dL")[0] is True

    # Invalid creatinine values
    assert checker.check("creatinine", 0.05, "mg/dL")[0] is False
    assert checker.check("creatinine", 50.0, "mg/dL")[0] is False
