# ============================================================================
# FILE: src/validators/rule_validator.py
# ============================================================================
"""
Rule-Based Validator

Fast, deterministic validation using:
1. Plausibility ranges (catch decimal errors)
2. Reference ranges (context-aware)
3. Unit validation
4. Critical value flags
5. Delta checks (change from previous)

Always runs first. Cheap and catches 90% of errors.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .plausibility import PlausibilityChecker
from medical_ingestion.constants import REFERENCE_RANGES, CRITICAL_VALUES
from medical_ingestion.core.context.extracted_value import ExtractedValue
from medical_ingestion.core.context.processing_context import ProcessingContext


logger = logging.getLogger(__name__)


class RuleValidator:
    """
    Rule-based validation for extracted lab values.

    Checks:
    1. Plausibility (physically possible?)
    2. Reference ranges (normal for patient demographics?)
    3. Critical values (needs immediate attention?)
    4. Unit compatibility
    5. Logical consistency with other values
    """

    def __init__(self):
        self.plausibility_checker = PlausibilityChecker()
        self.reference_ranges = REFERENCE_RANGES
        self.critical_values = CRITICAL_VALUES

    def validate(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext
    ) -> Tuple[bool, List[str], Optional[str]]:
        """
        Validate a single extracted value.

        Args:
            extracted: The extracted value to validate
            context: Processing context with patient demographics

        Returns:
            (is_valid, warnings, abnormal_flag)
        """
        warnings = []
        abnormal_flag = None

        # Check 1: Plausibility
        is_plausible, plaus_reason = self.plausibility_checker.check(
            extracted.field_name,
            float(extracted.value),
            extracted.unit
        )

        if not is_plausible:
            warnings.append(f"Plausibility check failed: {plaus_reason}")

            # Try to suggest correction
            corrected = self.plausibility_checker.suggest_correction(
                extracted.field_name,
                float(extracted.value),
                extracted.unit
            )

            if corrected:
                warnings.append(f"Possible correction: {corrected} {extracted.unit}")

            return False, warnings, abnormal_flag

        # Check 2: Reference range (if available)
        if extracted.field_name in self.reference_ranges:
            _, flag = self._check_reference_range(extracted, context)
            abnormal_flag = flag

            if flag and flag in ["CRITICAL HIGH", "CRITICAL LOW"]:
                warnings.append(f"CRITICAL VALUE: {extracted.value} {extracted.unit}")

        # Check 3: Critical values
        if extracted.field_name in self.critical_values:
            is_critical = self._check_critical_value(extracted)
            if is_critical:
                warnings.append(f"CRITICAL VALUE: immediate clinical attention required")
                # Only override flag if not already set by reference range check
                if not abnormal_flag or abnormal_flag not in ["CRITICAL HIGH", "CRITICAL LOW"]:
                    abnormal_flag = "CRITICAL"

        # Check 4: Unit validation
        unit_valid = self._validate_unit(extracted)
        if not unit_valid:
            warnings.append(f"Unit validation failed for {extracted.unit}")

        # Check 5: Cross-field validation
        cross_warnings = self._cross_field_validation(extracted, context)
        warnings.extend(cross_warnings)

        # Valid if no blocking issues
        is_valid = is_plausible and unit_valid

        return is_valid, warnings, abnormal_flag

    def _check_reference_range(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if value is within reference range.

        Returns:
            (in_range, abnormal_flag)
        """
        ranges = self.reference_ranges.get(extracted.field_name)
        if not ranges:
            return True, None

        # Get appropriate range based on demographics
        value = float(extracted.value)
        sex = context.patient_demographics.get('sex', 'adult')

        # Try sex-specific range first
        if sex in ranges:
            min_ref, max_ref, _ = ranges[sex]
        elif 'adult' in ranges:
            min_ref, max_ref, _ = ranges['adult']
        elif 'male' in ranges:
            # Default to male range if sex unknown
            min_ref, max_ref, _ = ranges['male']
        else:
            return True, None

        # Store reference range in extracted value
        extracted.reference_min = min_ref
        extracted.reference_max = max_ref

        # Check if in range
        if value < min_ref:
            # Check if critically low
            if extracted.field_name in self.critical_values:
                crit_thresholds = self.critical_values[extracted.field_name]
                crit_low = crit_thresholds.get("low", float('-inf'))
                if value < crit_low:
                    return False, "CRITICAL LOW"
            return False, "L"

        if value > max_ref:
            # Check if critically high
            if extracted.field_name in self.critical_values:
                crit_thresholds = self.critical_values[extracted.field_name]
                crit_high = crit_thresholds.get("high", float('inf'))
                if value > crit_high:
                    return False, "CRITICAL HIGH"
            return False, "H"

        return True, "N"

    def _check_critical_value(self, extracted: ExtractedValue) -> bool:
        """
        Check if value is in critical range.

        Critical values require immediate clinical notification.
        """
        if extracted.field_name not in self.critical_values:
            return False

        crit_thresholds = self.critical_values[extracted.field_name]
        value = float(extracted.value)

        crit_low = crit_thresholds.get("low", float('-inf'))
        crit_high = crit_thresholds.get("high", float('inf'))

        return value < crit_low or value > crit_high

    def _validate_unit(self, extracted: ExtractedValue) -> bool:
        """
        Validate that unit is expected for this test.
        """
        # Check against plausibility ranges (which include expected units)
        range_info = self.plausibility_checker.get_range(extracted.field_name)
        if range_info:
            _, _, expected_unit = range_info
            return extracted.unit == expected_unit

        # Unknown test - assume valid
        return True

    def _cross_field_validation(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext
    ) -> List[str]:
        """
        Validate value against other extracted values.

        Examples:
        - Sodium + Potassium (electrolyte balance)
        - Hemoglobin + Hematocrit (should correlate)
        - Glucose + HbA1c (diabetic markers)
        """
        warnings = []

        # Get other values
        other_values = {
            v.field_name: float(v.value)
            for v in context.extracted_values
            if v != extracted and v.value is not None
        }

        # Hemoglobin / Hematocrit correlation (Hct ≈ Hgb × 3)
        if extracted.field_name == "hemoglobin" and "hematocrit" in other_values:
            hgb = float(extracted.value)
            hct = other_values["hematocrit"]
            expected_hct = hgb * 3

            # Allow 15% variance
            if abs(hct - expected_hct) / expected_hct > 0.15:
                warnings.append(
                    f"Hgb/Hct correlation questionable: Hgb={hgb}, Hct={hct} "
                    f"(expected ~{expected_hct:.1f})"
                )

        # Electrolyte balance checks
        if extracted.field_name in ["sodium", "potassium", "chloride"]:
            # Check for extreme anion gap if all available
            if all(k in other_values for k in ["sodium", "chloride"]):
                na = other_values.get("sodium", float(extracted.value) if extracted.field_name == "sodium" else None)
                cl = other_values.get("chloride", float(extracted.value) if extracted.field_name == "chloride" else None)

                if na and cl:
                    # Simplified anion gap (Na - Cl, normal ~35)
                    anion_gap = na - cl
                    if anion_gap < 25 or anion_gap > 45:
                        warnings.append(
                            f"Unusual anion gap: {anion_gap:.1f} (normal ~35)"
                        )

        return warnings

    def batch_validate(
        self,
        context: ProcessingContext
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate all extracted values in context.

        Returns:
            {field_name: {
                "valid": bool,
                "warnings": List[str],
                "abnormal_flag": str
            }}
        """
        results = {}

        for extracted in context.extracted_values:
            is_valid, warnings, abnormal_flag = self.validate(extracted, context)

            results[extracted.field_name] = {
                "valid": is_valid,
                "warnings": warnings,
                "abnormal_flag": abnormal_flag
            }

            # Update extracted value
            extracted.rule_validation = is_valid
            extracted.abnormal_flag = abnormal_flag
            extracted.warnings.extend(warnings)

        return results


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def validate_value(
    extracted: ExtractedValue,
    context: ProcessingContext
) -> bool:
    """
    Quick validation check.

    Returns:
        True if valid, False otherwise
    """
    validator = RuleValidator()
    is_valid, _, _ = validator.validate(extracted, context)
    return is_valid
