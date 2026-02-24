# ============================================================================
# FILE: src/validators/plausibility.py
# ============================================================================
"""
Plausibility Checks

Catches extreme errors (decimal point mistakes, unit errors).
Different from reference ranges - these are "physically possible" boundaries.

Example:
- Hemoglobin 142 g/dL → FAIL (likely meant 14.2)
- Hemoglobin 6.5 g/dL → PASS (low but possible)
"""

from typing import Tuple, Optional
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_ingestion.constants import PLAUSIBILITY_RANGES


logger = logging.getLogger(__name__)


class PlausibilityChecker:
    """
    Check if lab values are within plausible ranges.

    Plausibility ranges are WIDER than reference ranges.
    They catch obvious errors like:
    - Decimal point mistakes (14.2 → 142)
    - Unit conversion errors (mmol → mg)
    - Data entry errors (4.2 → 42)
    """

    def __init__(self):
        self.ranges = PLAUSIBILITY_RANGES

    def check(
        self,
        field_name: str,
        value: float,
        unit: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if value is plausible.

        Args:
            field_name: Lab test name (e.g. "hemoglobin")
            value: Numeric value
            unit: Unit of measurement

        Returns:
            (is_plausible, reason_if_not)
        """
        # Unknown tests are assumed plausible
        if field_name not in self.ranges:
            return True, None

        min_val, max_val, expected_unit = self.ranges[field_name]

        # Check unit match
        if unit != expected_unit:
            logger.warning(
                f"{field_name}: unit mismatch - expected {expected_unit}, got {unit}"
            )
            return False, f"Unit mismatch: expected {expected_unit}, got {unit}"

        # Check value range
        if value < min_val:
            reason = f"Value {value} below plausible minimum {min_val} {unit}"
            logger.warning(f"{field_name}: {reason}")
            return False, reason

        if value > max_val:
            reason = f"Value {value} above plausible maximum {max_val} {unit}"
            logger.warning(f"{field_name}: {reason}")
            return False, reason

        return True, None

    def suggest_correction(
        self,
        field_name: str,
        value: float,
        unit: str
    ) -> Optional[float]:
        """
        Suggest corrected value if common error detected.

        Common errors:
        - Decimal point shift: 142.0 → 14.2
        - Unit conversion: 42 mmol/L → 4.2 mmol/L

        Returns:
            Suggested corrected value, or None if no correction found
        """
        if field_name not in self.ranges:
            return None

        min_val, max_val, expected_unit = self.ranges[field_name]

        # Already plausible
        if min_val <= value <= max_val:
            return None

        # Try decimal shift right (142 → 14.2)
        if value > max_val:
            corrected = value / 10
            if min_val <= corrected <= max_val:
                logger.info(
                    f"{field_name}: suggesting decimal correction {value} → {corrected}"
                )
                return corrected

            # Try double shift (1420 → 14.2)
            corrected = value / 100
            if min_val <= corrected <= max_val:
                logger.info(
                    f"{field_name}: suggesting decimal correction {value} → {corrected}"
                )
                return corrected

        # Try decimal shift left (1.42 → 14.2)
        if value < min_val:
            corrected = value * 10
            if min_val <= corrected <= max_val:
                logger.info(
                    f"{field_name}: suggesting decimal correction {value} → {corrected}"
                )
                return corrected

        return None

    def batch_check(self, values: dict) -> dict:
        """
        Check multiple values at once.

        Args:
            values: {field_name: (value, unit), ...}

        Returns:
            {field_name: (is_plausible, reason), ...}
        """
        results = {}

        for field_name, (value, unit) in values.items():
            is_plausible, reason = self.check(field_name, value, unit)
            results[field_name] = (is_plausible, reason)

        return results

    def get_range(self, field_name: str) -> Optional[Tuple[float, float, str]]:
        """
        Get plausibility range for a test.

        Returns:
            (min, max, unit) or None if unknown
        """
        return self.ranges.get(field_name)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_plausibility(field_name: str, value: float, unit: str) -> bool:
    """
    Quick plausibility check.

    Returns:
        True if plausible, False otherwise
    """
    checker = PlausibilityChecker()
    is_plausible, _ = checker.check(field_name, value, unit)
    return is_plausible


def get_plausibility_range(field_name: str) -> Optional[Tuple[float, float, str]]:
    """
    Get plausibility range for a lab test.

    Returns:
        (min, max, unit) or None
    """
    checker = PlausibilityChecker()
    return checker.get_range(field_name)
