# ============================================================================
# src/medical_ingestion/enrichers/lab_enricher.py
# ============================================================================
"""
Lab Enricher

Enriches lab test results with:
- LOINC codes (via lab_test_db.py — full 96K LOINC database)
- Standard reference ranges (region-aware, from loinc_mappings.json)
- Abnormal flag calculation
- Critical value detection
- OCR correction for test names
- validation_status for regulatory audit trail
"""

import logging
import re
from typing import Dict, Any, Optional, List

from .base import TypeSpecificEnricher, EnrichedExtraction

logger = logging.getLogger(__name__)


# Fallback reference ranges (adult) — used when lab_test_db is unavailable
_FALLBACK_REFERENCE_RANGES = {
    "718-7": {"min": 12.0, "max": 17.5, "unit": "g/dL", "name": "Hemoglobin"},
    "4544-3": {"min": 36.0, "max": 50.0, "unit": "%", "name": "Hematocrit"},
    "6690-2": {"min": 4.5, "max": 11.0, "unit": "K/uL", "name": "WBC"},
    "789-8": {"min": 4.0, "max": 5.5, "unit": "M/uL", "name": "RBC"},
    "777-3": {"min": 150, "max": 400, "unit": "K/uL", "name": "Platelet"},
    "2345-7": {"min": 70, "max": 100, "unit": "mg/dL", "name": "Glucose"},
    "3094-0": {"min": 7, "max": 20, "unit": "mg/dL", "name": "BUN"},
    "2160-0": {"min": 0.7, "max": 1.3, "unit": "mg/dL", "name": "Creatinine"},
    "2951-2": {"min": 136, "max": 145, "unit": "mEq/L", "name": "Sodium"},
    "2823-3": {"min": 3.5, "max": 5.0, "unit": "mEq/L", "name": "Potassium"},
    "2075-0": {"min": 98, "max": 106, "unit": "mEq/L", "name": "Chloride"},
    "1920-8": {"min": 10, "max": 40, "unit": "U/L", "name": "AST"},
    "1742-6": {"min": 7, "max": 56, "unit": "U/L", "name": "ALT"},
    "2093-3": {"min": 0, "max": 200, "unit": "mg/dL", "name": "Total Cholesterol"},
    "3016-3": {"min": 0.4, "max": 4.0, "unit": "mIU/L", "name": "TSH"},
}

# Fallback critical values — used when lab_test_db is unavailable
_FALLBACK_CRITICAL_VALUES = {
    "718-7": {"low": 7.0, "high": 20.0},   # Hemoglobin
    "2823-3": {"low": 2.5, "high": 6.5},   # Potassium
    "2951-2": {"low": 120, "high": 160},    # Sodium
    "2345-7": {"low": 50, "high": 400},     # Glucose
    "777-3": {"low": 50, "high": 1000},     # Platelet
}


class LabEnricher(TypeSpecificEnricher):
    """
    Enriches lab test results with LOINC codes, reference ranges,
    and abnormal flag calculations.

    Uses the full LOINC database via lab_test_db.py for:
    - Exact name matching
    - Fuzzy/alias matching (abbreviations, relatednames)
    - OCR-tolerant matching
    - Region-aware reference ranges

    Sets validation_status on each test result:
    - verified: Found in LOINC (exact/fuzzy/alias/prefix match)
    - ocr_corrected: Found via OCR edit distance correction
    - unverified: Not found in any database — needs human review
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._lab_db = None
        self._db_available = None

    def _get_lab_db(self):
        """Lazy-load lab test database."""
        if self._lab_db is None:
            try:
                from ..constants.lab_test_db import get_lab_test_db
                self._lab_db = get_lab_test_db()
                self._db_available = self._lab_db.is_available
            except Exception as e:
                logger.warning(f"Could not load lab test database: {e}")
                self._lab_db = None
                self._db_available = False
        return self._lab_db

    @property
    def enricher_type(self) -> str:
        return "lab"

    async def enrich(self, extraction: Any) -> EnrichedExtraction:
        """
        Enrich lab test results.

        Adds:
        - LOINC codes (from full database)
        - Standard reference ranges
        - Calculated abnormal flags
        - Critical value alerts
        - OCR corrections
        - validation_status for each test
        """
        result = self._create_result(extraction)
        enrichments = {
            "loinc_codes": [],
            "reference_ranges_added": 0,
            "abnormal_flags_calculated": 0,
            "critical_values": [],
            "ocr_corrections": [],
            "validation_summary": {}
        }

        validation_warnings = []
        review_reasons = []
        status_counts = {}

        # Process each test result
        for test_result in extraction.test_results:
            enrichment = self._enrich_test_result(test_result)

            # Track validation status
            vs = test_result.validation_status or 'unverified'
            status_counts[vs] = status_counts.get(vs, 0) + 1

            if enrichment.get("loinc_code"):
                enrichments["loinc_codes"].append({
                    "test": test_result.name,
                    "loinc": enrichment["loinc_code"],
                    "loinc_name": enrichment.get("loinc_name"),
                    "match_type": enrichment.get("match_type")
                })

            if enrichment.get("corrected_name"):
                enrichments["ocr_corrections"].append({
                    "original": enrichment.get("original_name", test_result.name),
                    "corrected": enrichment["corrected_name"]
                })

            if enrichment.get("reference_range_added"):
                enrichments["reference_ranges_added"] += 1

            if enrichment.get("abnormal_flag_calculated"):
                enrichments["abnormal_flags_calculated"] += 1

            if enrichment.get("is_critical"):
                enrichments["critical_values"].append({
                    "test": test_result.name,
                    "value": test_result.value,
                    "critical_range": enrichment.get("critical_range")
                })
                review_reasons.append(f"critical_value_{test_result.name}")

            if vs == 'unverified':
                validation_warnings.append(
                    f"Lab test '{test_result.name}' could not be verified against LOINC database"
                )
                review_reasons.append(f"unverified_lab_test_{test_result.name}")

        enrichments["validation_summary"] = status_counts

        # Calculate enrichment confidence
        total_tests = len(extraction.test_results)
        verified_count = sum(1 for t in extraction.test_results
                            if t.validation_status in ('verified', 'ocr_corrected'))
        confidence = verified_count / total_tests if total_tests > 0 else 0.5

        result.enrichments = enrichments
        result.enrichment_confidence = confidence
        result.validation_warnings = validation_warnings
        result.requires_review = len(review_reasons) > 0
        result.review_reasons = review_reasons

        logger.info(
            f"Lab enrichment complete: "
            f"{' | '.join(f'{c} {s}' for s, c in sorted(status_counts.items()))}, "
            f"{len(enrichments['critical_values'])} critical values"
        )

        return result

    def _enrich_test_result(self, test_result: Any) -> Dict[str, Any]:
        """Enrich a single test result using the full LOINC database."""
        enrichment = {}
        original_name = test_result.name

        # Try the real LOINC database first
        lab_db = self._get_lab_db()
        if lab_db and self._db_available:
            from ..constants.lab_test_db import lookup_lab_test

            db_result = lookup_lab_test(
                original_name,
                ocr_correction=True,
                expected_unit=test_result.unit
            )

            if db_result:
                loinc_code = db_result.get('loinc_num')
                match_type = db_result.get('match_type', 'exact')

                test_result.loinc_code = loinc_code
                test_result.loinc_name = db_result.get('component')

                enrichment["loinc_code"] = loinc_code
                enrichment["loinc_name"] = db_result.get('component')
                enrichment["match_type"] = match_type

                # Determine validation_status based on match type
                if match_type == 'ocr_corrected':
                    test_result.validation_status = 'ocr_corrected'
                    enrichment["corrected_name"] = db_result.get('component')
                    enrichment["original_name"] = original_name
                    enrichment["edit_distance"] = db_result.get('edit_distance')
                else:
                    test_result.validation_status = 'verified'

                logger.debug(
                    f"LOINC match: '{original_name}' -> "
                    f"{loinc_code} ({match_type}, status: {test_result.validation_status})"
                )

                # Use database reference ranges
                ref_range = lab_db.get_reference_range(loinc_code)
                if ref_range and not test_result.reference_range:
                    low = ref_range.get('low')
                    high = ref_range.get('high')
                    if low is not None and high is not None:
                        test_result.reference_range = f"{low}-{high}"
                        enrichment["reference_range_added"] = True

                # Use database critical values
                critical_vals = lab_db.get_critical_values(loinc_code)
                if critical_vals and test_result.value is not None:
                    critical = self._check_critical_value_from_dict(
                        test_result.value, critical_vals
                    )
                    if critical:
                        enrichment["is_critical"] = True
                        enrichment["critical_range"] = critical_vals

                # Calculate abnormal flag if not present
                if not test_result.abnormal_flag and test_result.value is not None:
                    flag = self._calculate_abnormal_flag(test_result, loinc_code)
                    if flag:
                        test_result.abnormal_flag = flag
                        enrichment["abnormal_flag_calculated"] = True

                return enrichment

        # Database unavailable — fall back to hardcoded mappings
        loinc_code = self._lookup_loinc_fallback(original_name.lower().strip())
        if loinc_code:
            test_result.loinc_code = loinc_code
            test_result.validation_status = 'verified'
            enrichment["loinc_code"] = loinc_code
            enrichment["match_type"] = "fallback_dict"

            # Fallback reference ranges
            if not test_result.reference_range:
                ref_range = _FALLBACK_REFERENCE_RANGES.get(loinc_code)
                if ref_range:
                    test_result.reference_range = f"{ref_range['min']}-{ref_range['max']}"
                    enrichment["reference_range_added"] = True

            # Fallback critical values
            if test_result.value is not None:
                critical = self._check_critical_value_fallback(test_result.value, loinc_code)
                if critical:
                    enrichment["is_critical"] = True
                    enrichment["critical_range"] = critical

            # Calculate abnormal flag
            if not test_result.abnormal_flag and test_result.value is not None:
                flag = self._calculate_abnormal_flag(test_result, loinc_code)
                if flag:
                    test_result.abnormal_flag = flag
                    enrichment["abnormal_flag_calculated"] = True

            return enrichment

        # No match anywhere — mark as unverified
        test_result.validation_status = 'unverified'
        logger.warning(
            f"UNVERIFIED lab test: '{original_name}' — not found in LOINC database"
        )
        return enrichment

    def _lookup_loinc_fallback(self, test_name: str) -> Optional[str]:
        """Fallback LOINC lookup using hardcoded common test mappings."""
        # These cover ~86 common test name variants
        FALLBACK_MAPPINGS = {
            # Hematology
            "hemoglobin": "718-7", "hgb": "718-7", "hb": "718-7",
            "hematocrit": "4544-3", "hct": "4544-3",
            "wbc": "6690-2", "white blood cell": "6690-2",
            "rbc": "789-8", "red blood cell": "789-8",
            "platelet": "777-3", "plt": "777-3", "platelets": "777-3",
            "mcv": "787-2", "mch": "785-6", "mchc": "786-4", "rdw": "788-0",
            # Chemistry
            "glucose": "2345-7", "bun": "3094-0", "blood urea nitrogen": "3094-0",
            "creatinine": "2160-0", "sodium": "2951-2", "na": "2951-2",
            "potassium": "2823-3", "k": "2823-3", "chloride": "2075-0", "cl": "2075-0",
            "co2": "2028-9", "bicarbonate": "2028-9",
            "calcium": "17861-6", "ca": "17861-6",
            "magnesium": "2601-3", "mg": "2601-3", "phosphorus": "2777-1",
            # Liver
            "ast": "1920-8", "sgot": "1920-8", "alt": "1742-6", "sgpt": "1742-6",
            "alkaline phosphatase": "6768-6", "alp": "6768-6",
            "bilirubin": "1975-2", "total bilirubin": "1975-2",
            "albumin": "1751-7", "total protein": "2885-2",
            # Lipids
            "cholesterol": "2093-3", "total cholesterol": "2093-3",
            "cholesterol total": "2093-3", "cholesterol, total": "2093-3",
            "triglycerides": "2571-8", "hdl": "2085-9", "ldl": "2089-1",
            "non-hdl cholesterol": "43396-1", "non hdl cholesterol": "43396-1",
            "chol/hdlc ratio": "9830-1",
            # Thyroid
            "tsh": "3016-3", "t4": "3026-2", "free t4": "3024-7", "t3": "3053-6",
            # Coagulation
            "pt": "5902-2", "prothrombin time": "5902-2",
            "inr": "6301-6", "ptt": "3173-2", "aptt": "3173-2",
            # Cardiac
            "troponin": "10839-9", "bnp": "30934-4", "ck": "2157-6", "ck-mb": "13969-1",
            # Other
            "hba1c": "4548-4", "hemoglobin a1c": "4548-4",
            "esr": "4537-7", "sed rate": "4537-7", "sedimentation rate": "4537-7",
            "crp": "1988-5", "c-reactive protein": "1988-5",
            "mpv": "32623-1", "mean platelet volume": "32623-1",
            "vitamin d": "1989-3", "vitamin d, 25-oh": "1989-3",
            # eGFR
            "egfr": "48642-3", "egfr non-afr. american": "48642-3",
            "egfr african american": "48643-1",
        }

        # Direct match
        if test_name in FALLBACK_MAPPINGS:
            return FALLBACK_MAPPINGS[test_name]

        # Partial match
        for key, loinc in FALLBACK_MAPPINGS.items():
            if key in test_name or test_name in key:
                return loinc

        return None

    def _calculate_abnormal_flag(
        self,
        test_result: Any,
        loinc_code: Optional[str]
    ) -> Optional[str]:
        """Calculate abnormal flag based on value and reference range."""
        try:
            value = float(test_result.value)
        except (ValueError, TypeError):
            return None

        # Try to use reference range from test result
        if test_result.reference_range:
            ref_min, ref_max = self._parse_reference_range(test_result.reference_range)
            if ref_min is not None and ref_max is not None:
                if value < ref_min:
                    return "L"
                elif value > ref_max:
                    return "H"
                return None

        # Fall back to hardcoded reference ranges
        if loinc_code and loinc_code in _FALLBACK_REFERENCE_RANGES:
            ref = _FALLBACK_REFERENCE_RANGES[loinc_code]
            if value < ref["min"]:
                return "L"
            elif value > ref["max"]:
                return "H"

        return None

    def _parse_reference_range(self, ref_range: str) -> tuple:
        """Parse reference range string to min/max values."""
        # Handle "10-20" format
        match = re.match(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', ref_range)
        if match:
            return float(match.group(1)), float(match.group(2))

        # Handle "< 100" format
        match = re.match(r'<\s*(\d+\.?\d*)', ref_range)
        if match:
            return 0, float(match.group(1))

        # Handle "> 50" format
        match = re.match(r'>\s*(\d+\.?\d*)', ref_range)
        if match:
            return float(match.group(1)), float('inf')

        return None, None

    def _check_critical_value_from_dict(
        self,
        value: Any,
        critical_vals: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Check if value is in critical range using a dict with low/high keys."""
        try:
            value = float(value)
        except (ValueError, TypeError):
            return None

        low = critical_vals.get('low')
        high = critical_vals.get('high')
        if low is not None and value < low:
            return critical_vals
        if high is not None and value > high:
            return critical_vals

        return None

    def _check_critical_value_fallback(
        self,
        value: Any,
        loinc_code: str
    ) -> Optional[Dict[str, float]]:
        """Check if value is in critical range using fallback table."""
        if loinc_code not in _FALLBACK_CRITICAL_VALUES:
            return None

        try:
            value = float(value)
        except (ValueError, TypeError):
            return None

        critical = _FALLBACK_CRITICAL_VALUES[loinc_code]

        if value < critical["low"] or value > critical["high"]:
            return critical

        return None
