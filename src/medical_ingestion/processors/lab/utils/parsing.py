# src/medical_ingestion/processors/lab/utils/parsing.py
"""
Parsing utilities for lab value extraction.
"""

import re
from typing import Any, Dict, Optional, Tuple


def parse_numeric_value(value_str: str) -> Optional[float]:
    """
    Extract numeric value from string.

    Handles values like:
    - "12.5"
    - "1024 High"  (value with embedded flag)
    - "< 0.5"      (less-than values)
    - "> 100"      (greater-than values)

    Rejects unit-like strings that would produce wrong values:
    - "x10E3/uL" should NOT return 103
    - "g/dL" should NOT return a number
    """
    if not value_str:
        return None

    value_str = value_str.strip()

    # Reject strings that look like units (would produce garbage numbers)
    # These patterns indicate a unit, not a value
    unit_patterns = [
        r'^x?10E\d',           # x10E3, 10E6 (scientific notation units)
        r'^[a-zA-Z]+/[a-zA-Z]+',  # g/dL, mg/dL, etc.
        r'^/[a-zA-Z]+',        # /uL, /dL (unit denominators)
        r'^\d*E\d+/',          # 10E3/uL
    ]
    for pattern in unit_patterns:
        if re.match(pattern, value_str, re.IGNORECASE):
            return None

    # First, try to find a numeric value at the start (handles "1024 High")
    match = re.match(r'^[<>]?\s*([\d.]+)', value_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Fallback: extract any numeric value, but only if it looks like a real value
    # Reject if the string has unit indicators mixed in
    if re.search(r'E\d+/', value_str, re.IGNORECASE):  # Scientific notation units
        return None

    cleaned = re.sub(r'[^\d.\-+]', '', value_str)
    # Sanity check: cleaned string should be reasonably short (not concatenated garbage)
    if len(cleaned) > 10:  # Real lab values rarely exceed 10 digits
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_value_and_flag(value_str: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse a combined value+flag string like "1024 High" or "44.5 High".

    LabCorp Enterprise Report format often has the flag embedded in the result column.

    Returns:
        Tuple of (numeric_value, flag) where flag may be None
    """
    if not value_str:
        return None, None

    value_str = value_str.strip()

    # Common flag patterns
    flag_patterns = [
        (r'\b(High)\b', 'H'),
        (r'\b(Low)\b', 'L'),
        (r'\bHH\b', 'HH'),
        (r'\bLL\b', 'LL'),
        (r'\bH\b', 'H'),
        (r'\bL\b', 'L'),
        (r'\bCRITICAL\b', 'CRITICAL'),
        (r'\*+$', '*'),
    ]

    flag = None
    clean_value = value_str

    # Extract flag if present
    for pattern, normalized in flag_patterns:
        match = re.search(pattern, value_str, re.IGNORECASE)
        if match:
            flag = normalized
            # Remove flag from value string
            clean_value = re.sub(pattern, '', value_str, flags=re.IGNORECASE).strip()
            break

    # Parse numeric value
    numeric_value = parse_numeric_value(clean_value)

    return numeric_value, flag


def parse_reference_range(ref_str: str) -> Optional[Tuple[float, float]]:
    """
    Parse reference range string into (low, high) tuple.

    Handles multiple formats:
    - "12.0-15.5" or "12.0 - 15.5" (standard range)
    - "4.5-11.0 x10E3/uL" (range with unit)
    - ">=10" or "> 5.0" (greater than)
    - "<=100" or "< 0.5" (less than)
    - "80-100" (integer range)
    - "0.0 - 10.0" (with decimal)
    - "Negative" or "Non-Reactive" (qualitative - return None)

    Returns:
        Tuple of (low, high) floats, or None if parsing fails
        For ">=X", returns (X, None-represented-as-inf)
        For "<=X", returns (None-represented-as-0, X)
    """
    if not ref_str:
        return None

    ref_str = ref_str.strip()

    # Skip qualitative results
    qualitative_patterns = [
        r'^negative$', r'^positive$', r'^non[\-\s]?reactive$',
        r'^reactive$', r'^normal$', r'^abnormal$', r'^see\s+', r'^n/a$'
    ]
    for pattern in qualitative_patterns:
        if re.match(pattern, ref_str, re.IGNORECASE):
            return None

    # Standard range format: "12.0-15.5" or "12.0 - 15.5"
    # Also handles ranges with units: "4.5-11.0 x10E3/uL"
    range_match = re.search(r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)', ref_str)
    if range_match:
        try:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            return (low, high)
        except ValueError:
            pass

    # Greater than format: ">=10" or ">5.0" or "> 5"
    gt_match = re.match(r'^[>≥]\s*=?\s*(\d+\.?\d*)', ref_str)
    if gt_match:
        try:
            low = float(gt_match.group(1))
            return (low, float('inf'))  # Using inf for "no upper limit"
        except ValueError:
            pass

    # Less than format: "<=100" or "<0.5" or "< 100"
    lt_match = re.match(r'^[<≤]\s*=?\s*(\d+\.?\d*)', ref_str)
    if lt_match:
        try:
            high = float(lt_match.group(1))
            return (0.0, high)  # Using 0 for "no lower limit"
        except ValueError:
            pass

    return None


def extract_reference_range(ref_str: str) -> Optional[Dict[str, Any]]:
    """
    Extract reference range from string.

    Returns dict with 'low' and 'high' keys, or None if parsing fails.
    """
    result = parse_reference_range(ref_str)
    if result:
        return {'low': result[0], 'high': result[1]}
    return None


def parse_lab_value(value_str: str) -> Dict[str, Any]:
    """
    Parse a lab value string into value and unit components.

    Handles formats like:
    - "12.5 mg/dL"
    - "1024 High"
    - "< 0.5 ng/mL"
    - "45.2"

    Returns:
        Dict with 'value' (float or None), 'unit' (str), 'flag' (str or None)
    """
    if not value_str:
        return {'value': None, 'unit': '', 'flag': None}

    value_str = value_str.strip()

    # Parse value and flag together
    numeric_value, flag = parse_value_and_flag(value_str)

    # Extract unit - look for text after the number
    unit = ''
    # Remove the numeric part and flag to find unit
    remaining = re.sub(r'^[<>]?\s*[\d.]+\s*', '', value_str)
    # Remove common flags
    remaining = re.sub(r'\b(High|Low|HH|LL|H|L|CRITICAL)\b', '', remaining, flags=re.IGNORECASE)
    remaining = remaining.strip()

    # Check if remaining looks like a unit
    if remaining and not remaining.isdigit():
        unit = remaining

    return {
        'value': numeric_value,
        'unit': unit,
        'flag': flag
    }


def is_likely_lab_test(name: str) -> bool:
    """
    Determine if a string is likely a lab test name.

    Returns True for strings that look like lab test names,
    False for headers, empty strings, or other non-test content.
    """
    if not name or len(name.strip()) < 2:
        return False

    name = name.strip()

    # Reject if it's a common header/non-test pattern
    non_test_patterns = [
        r'^(test|result|value|unit|reference|range|flag|status|date|time)s?$',
        r'^(patient|name|dob|id|mrn|specimen|collected|received)$',
        r'^(page|of|\d+)$',
        r'^\s*$',
        r'^-+$',
        r'^=+$',
    ]

    name_lower = name.lower()
    for pattern in non_test_patterns:
        if re.match(pattern, name_lower, re.IGNORECASE):
            return False

    # Accept if it matches common lab test patterns
    lab_test_patterns = [
        r'\b(wbc|rbc|hgb|hct|mcv|mch|mchc|plt|rdw)\b',
        r'\b(glucose|sodium|potassium|chloride|creatinine|bun)\b',
        r'\b(ast|alt|alp|ggt|bilirubin|albumin)\b',
        r'\b(tsh|t3|t4|hba1c|a1c)\b',
        r'\b(neutrophil|lymphocyte|monocyte|eosinophil|basophil)\b',
        r'\b(cholesterol|triglyceride|hdl|ldl)\b',
        r'\b(cd4|cd8|ratio)\b',
        r'\b(count|level|panel|screen)\b',
    ]

    for pattern in lab_test_patterns:
        if re.search(pattern, name_lower):
            return True

    # Accept if it has a reasonable length and contains letters
    if 2 <= len(name) <= 50 and re.search(r'[a-zA-Z]', name):
        return True

    return False


def _normalize_test_name(name: str) -> str:
    """Normalize test name for matching (remove special chars, collapse spaces)."""
    # Replace common separators with space
    normalized = re.sub(r'[/:,\-\(\)]', ' ', name.lower())
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def normalize_test_name(name: str) -> str:
    """Public alias for test name normalization."""
    return _normalize_test_name(name)


def find_value_in_table(
    table,
    field_name: str,
    value_column_index: int = 1
) -> Optional[Tuple[str, int, int]]:
    """
    Find a value in a table by field name.

    Uses normalized matching to handle variations like:
    - "CD4/CD8 Ratio" vs "CD4:CD8 Ratio" vs "CD 4/CD 8 Ratio"

    Returns:
        Tuple of (value_str, row_idx, col_idx) or None
    """
    field_normalized = _normalize_test_name(field_name)

    for row_idx, row in enumerate(table.rows):
        if not row:
            continue

        cell_value = row[0]
        cell_normalized = _normalize_test_name(cell_value)

        # Try exact match first (normalized)
        if field_normalized == cell_normalized:
            if value_column_index < len(row):
                value_str = row[value_column_index].strip()
                return (value_str, row_idx, value_column_index)

        # Try substring match (original logic)
        if field_name.lower() in cell_value.lower():
            if value_column_index < len(row):
                value_str = row[value_column_index].strip()
                return (value_str, row_idx, value_column_index)

        # Try normalized substring match
        if field_normalized in cell_normalized:
            if value_column_index < len(row):
                value_str = row[value_column_index].strip()
                return (value_str, row_idx, value_column_index)

    return None


def extract_abnormal_flag(
    table,
    row_idx: int,
    flag_column: int
) -> Optional[str]:
    """Extract abnormal flag from table."""
    if row_idx >= len(table.rows):
        return None

    row = table.rows[row_idx]
    if flag_column >= len(row):
        return None

    flag = row[flag_column].strip()

    # Only return if it's a valid flag
    if flag in ['H', 'L', 'HH', 'LL', 'CRITICAL', '*']:
        return flag

    return None


def correct_lab_value_ocr(value_str: str) -> str:
    """
    Fix common OCR/PDF text extraction errors in lab values.

    Common confusions in numeric contexts:
    - 'E' between/adjacent to digits → removed: "4E.9" → "4.9", "7E5" → "75"
    - 'O' between digits → '0': "1O5" → "105"
    - 'l'/'I' between digits → '1': "l2.5" → "12.5"
    - 'S' at start of number → '5': "S00" → "500"

    Also handles reference range strings like "7-2O5" → "7-25".

    Returns original string if no correction needed or if correction
    produces an invalid result.
    """
    if not value_str or not isinstance(value_str, str):
        return value_str or ''

    original = value_str.strip()

    # Skip if no digits present (purely qualitative like "Negative")
    if not re.search(r'\d', original):
        return original

    # For reference range strings (contain '-' or '–'), correct each part separately
    if re.search(r'\d\s*[-–]\s*\d', original):
        parts = re.split(r'([-–])', original, maxsplit=1)
        if len(parts) == 3:
            left = _correct_numeric_ocr(parts[0].strip())
            right = _correct_numeric_ocr(parts[2].strip())
            return f"{left}{parts[1]}{right}"

    return _correct_numeric_ocr(original)


def _correct_numeric_ocr(token: str) -> str:
    """
    Apply OCR corrections to a single numeric token.

    Strategy: try removing/replacing suspicious alpha chars between digits.
    Only accept the correction if the result parses as a valid number.
    """
    if not token or not re.search(r'\d', token):
        return token

    corrected = token

    # Rule 1: Remove stray 'E' between/adjacent to digits (NOT scientific notation like "10E3/uL")
    # "4E.9" → "4.9", "7E5" → "75"
    # But preserve legitimate "E" in unit strings like "x10E3"
    if 'E' in corrected or 'e' in corrected:
        # Only fix if this doesn't look like scientific notation with a unit
        if not re.search(r'10E\d+/', corrected, re.IGNORECASE):
            corrected = re.sub(r'(\d)[Ee](\d)', r'\1\2', corrected)
            corrected = re.sub(r'(\d)[Ee](\.)', r'\1\2', corrected)

    # Rule 2: Replace 'O' with '0' when between digits
    corrected = re.sub(r'(\d)[Oo](\d)', r'\g<1>0\2', corrected)

    # Rule 3: Replace 'l' or 'I' with '1' when adjacent to digits
    corrected = re.sub(r'(\d)[lI](\d)', r'\g<1>1\2', corrected)
    # Also at start of string: "l2.5" → "12.5"
    corrected = re.sub(r'^[lI](\d)', r'1\1', corrected)

    # Rule 4: Replace 'S' with '5' at start when followed by digits
    corrected = re.sub(r'^S(\d)', r'5\1', corrected)

    # Rule 5: Replace 'B' with '8' when between digits
    corrected = re.sub(r'(\d)B(\d)', r'\g<1>8\2', corrected)

    # Validate: corrected should be parseable as a number (possibly with unit suffix)
    if corrected != token:
        # Extract just the numeric part for validation
        num_match = re.match(r'^[<>]?\s*([\d.]+)', corrected)
        if num_match:
            try:
                float(num_match.group(1))
                return corrected
            except ValueError:
                return token  # Revert if invalid
        # If no leading number, check if it's at least not worse
        return corrected

    return token
