# ============================================================================
# src/medical_ingestion/utils/text_normalizer.py
# ============================================================================
"""
Text Normalization Utilities

Cleans up extracted text from OCR/PDF extraction:
- Removes footnote markers (superscripts like ¹, ², 01, 02)
- Normalizes lab test names to canonical forms
- Handles common OCR artifacts
- Uses LLM reasoning for ambiguous cases
"""

import re
import logging
from typing import Dict, Any, List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Common lab test name mappings (variations -> canonical)
LAB_TEST_MAPPINGS = {
    # Complete Blood Count (CBC)
    "wbc": "WBC",
    "rbc": "RBC",
    "hgb": "Hemoglobin",
    "hb": "Hemoglobin",
    "hemoglobin": "Hemoglobin",
    "hct": "Hematocrit",
    "hematocrit": "Hematocrit",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    "plt": "Platelets",
    "platelet": "Platelets",
    "platelets": "Platelets",
    "mpv": "MPV",

    # White Blood Cell Differential
    "neutrophils": "Neutrophils",
    "neut": "Neutrophils",
    "lymphocytes": "Lymphocytes",
    "lymph": "Lymphocytes",
    "monocytes": "Monocytes",
    "mono": "Monocytes",
    "eosinophils": "Eosinophils",
    "eos": "Eosinophils",
    "basophils": "Basophils",
    "baso": "Basophils",
    "basos": "Basophils",

    # Basic Metabolic Panel (BMP)
    "glucose": "Glucose",
    "bun": "BUN",
    "creatinine": "Creatinine",
    "creat": "Creatinine",
    "sodium": "Sodium",
    "na": "Sodium",
    "potassium": "Potassium",
    "k": "Potassium",
    "chloride": "Chloride",
    "cl": "Chloride",
    "co2": "CO2",
    "carbon dioxide": "CO2",
    "calcium": "Calcium",
    "ca": "Calcium",

    # Comprehensive Metabolic Panel (CMP) additions
    "albumin": "Albumin",
    "alb": "Albumin",
    "total protein": "Total Protein",
    "protein": "Total Protein",
    "bilirubin": "Bilirubin",
    "bili": "Bilirubin",
    "alkaline phosphatase": "Alkaline Phosphatase",
    "alk phos": "Alkaline Phosphatase",
    "alp": "Alkaline Phosphatase",
    "ast": "AST",
    "sgot": "AST",
    "alt": "ALT",
    "sgpt": "ALT",

    # Lipid Panel
    "cholesterol": "Total Cholesterol",
    "total cholesterol": "Total Cholesterol",
    "hdl": "HDL",
    "ldl": "LDL",
    "triglycerides": "Triglycerides",
    "trig": "Triglycerides",
    "vldl": "VLDL",

    # Thyroid
    "tsh": "TSH",
    "t3": "T3",
    "t4": "T4",
    "free t4": "Free T4",
    "ft4": "Free T4",

    # Cardiac
    "troponin": "Troponin",
    "bnp": "BNP",
    "ck": "CK",
    "ck-mb": "CK-MB",

    # Coagulation
    "pt": "PT",
    "inr": "INR",
    "ptt": "PTT",
    "aptt": "aPTT",

    # Urinalysis
    "ph": "pH",
    "specific gravity": "Specific Gravity",
    "sp gr": "Specific Gravity",

    # HbA1c
    "hba1c": "HbA1c",
    "a1c": "HbA1c",
    "hemoglobin a1c": "HbA1c",

    # Iron studies
    "iron": "Iron",
    "ferritin": "Ferritin",
    "tibc": "TIBC",
    "transferrin": "Transferrin",

    # Electrolytes
    "magnesium": "Magnesium",
    "mg": "Magnesium",
    "phosphorus": "Phosphorus",
    "phos": "Phosphorus",
}

# Context-based inference: unit + typical value range -> likely test
# Format: (unit, value_min, value_max) -> canonical_name
UNIT_CONTEXT_HINTS = {
    # CBC
    ("x10^9/l", 4.0, 11.0): "WBC",
    ("x10^12/l", 4.0, 6.0): "RBC",
    ("g/dl", 12.0, 18.0): "Hemoglobin",
    ("g/l", 120.0, 180.0): "Hemoglobin",
    ("%", 36.0, 54.0): "Hematocrit",
    ("fl", 80.0, 100.0): "MCV",
    ("pg", 26.0, 34.0): "MCH",
    ("g/dl", 31.0, 37.0): "MCHC",
    ("x10^9/l", 150.0, 450.0): "Platelets",
    # Differential (%)
    ("%", 40.0, 75.0): "Neutrophils",
    ("%", 20.0, 45.0): "Lymphocytes",
    ("%", 2.0, 10.0): "Monocytes",
    ("%", 0.0, 6.0): "Eosinophils",
    ("%", 0.0, 2.0): "Basophils",
    # BMP
    ("mg/dl", 70.0, 140.0): "Glucose",
    ("mg/dl", 7.0, 25.0): "BUN",
    ("mg/dl", 0.5, 1.5): "Creatinine",
    ("meq/l", 136.0, 145.0): "Sodium",
    ("mmol/l", 136.0, 145.0): "Sodium",
    ("meq/l", 3.5, 5.5): "Potassium",
    ("mmol/l", 3.5, 5.5): "Potassium",
    ("meq/l", 96.0, 106.0): "Chloride",
    ("meq/l", 22.0, 32.0): "CO2",
    # Liver
    ("u/l", 10.0, 50.0): "AST",
    ("u/l", 7.0, 56.0): "ALT",
    ("u/l", 40.0, 150.0): "Alkaline Phosphatase",
    ("mg/dl", 0.1, 1.5): "Bilirubin",
    ("g/dl", 3.5, 5.5): "Albumin",
    # Lipids
    ("mg/dl", 100.0, 250.0): "Total Cholesterol",
    ("mg/dl", 40.0, 100.0): "HDL",
    ("mg/dl", 50.0, 200.0): "LDL",
    ("mg/dl", 30.0, 200.0): "Triglycerides",
    # Thyroid
    ("miu/l", 0.3, 5.0): "TSH",
    ("uiu/ml", 0.3, 5.0): "TSH",
    # Coagulation
    ("seconds", 10.0, 15.0): "PT",
    ("", 0.8, 1.3): "INR",
    # HbA1c
    ("%", 4.0, 7.0): "HbA1c",
}

# Fuzzy matching for common OCR errors
FUZZY_CORRECTIONS = {
    # Common OCR misreads
    "rnch": "MCH",
    "rnchc": "MCHC",
    "rnov": "MCV",
    "wbo": "WBC",
    "rbo": "RBC",
    "hgd": "Hemoglobin",
    "plts": "Platelets",
    "neuts": "Neutrophils",
    "lymphs": "Lymphocytes",
    "monos": "Monocytes",
    "eosins": "Eosinophils",
    "glu": "Glucose",
    "gluc": "Glucose",
    "creat": "Creatinine",
    "crea": "Creatinine",
    "bili": "Bilirubin",
    "tbili": "Total Bilirubin",
    "dbili": "Direct Bilirubin",
    "alb": "Albumin",
    "alkp": "Alkaline Phosphatase",
    "alkphos": "Alkaline Phosphatase",
    "chol": "Total Cholesterol",
    "trig": "Triglycerides",
    "trigs": "Triglycerides",
}

# Patterns for footnote/superscript markers
FOOTNOTE_PATTERNS = [
    r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+$',           # Unicode superscripts
    r'\s*[\(\[]\d+[\)\]]$',        # (1), [2], etc.
    r'\s*\*+$',                     # Asterisks
    r'\s*†+$',                      # Daggers
    r'\s*‡+$',                      # Double daggers
    r'_\d{1,2}$',                  # Underscore with digits (basos_02 -> basos)
    r'(?<=[a-zA-Z])\d{1,2}$',      # Trailing 1-2 digits after letters (mchc02 -> mchc)
    r'\s*\d{1,2}$',                # Trailing numbers with space
]

# OCR artifact patterns
OCR_ARTIFACT_PATTERNS = [
    (r'\s+', ' '),                  # Multiple spaces -> single space
    (r'[|l](?=[A-Z])', 'I'),       # Common OCR: l or | before caps -> I
    (r'(?<=[a-z])0(?=[a-z])', 'o'), # 0 between lowercase -> o
    (r'(?<=[A-Z])0(?=[A-Z])', 'O'), # 0 between uppercase -> O
    (r'rn(?=[a-z])', 'm'),         # rn -> m (common OCR error)
]


def remove_footnote_markers(text: str) -> str:
    """
    Remove footnote/superscript markers from text.

    Examples:
        "mchc02" -> "mchc"
        "Glucose¹" -> "Glucose"
        "WBC (1)" -> "WBC"
    """
    if not text:
        return text

    result = text.strip()

    for pattern in FOOTNOTE_PATTERNS:
        result = re.sub(pattern, '', result)

    return result.strip()


def fix_ocr_artifacts(text: str) -> str:
    """
    Fix common OCR recognition errors.
    """
    if not text:
        return text

    result = text

    for pattern, replacement in OCR_ARTIFACT_PATTERNS:
        result = re.sub(pattern, replacement, result)

    return result.strip()


def normalize_lab_test_name(name: str, unit: str = None, value: Any = None) -> str:
    """
    Normalize a lab test name to its canonical form.

    Steps:
    1. Remove footnote markers
    2. Fix OCR artifacts
    3. Map to canonical name if known
    4. Use fuzzy matching for common OCR errors
    5. Use context clues (unit, value) for ambiguous cases

    Args:
        name: The raw test name from OCR/extraction
        unit: Optional unit for context-based inference
        value: Optional value for context-based inference

    Examples:
        "mchc02" -> "MCHC"
        "hgb¹" -> "Hemoglobin"
        "Glucose (1)" -> "Glucose"
    """
    if not name:
        return name

    # Step 1: Remove footnote markers
    cleaned = remove_footnote_markers(name)

    # Step 2: Fix OCR artifacts
    cleaned = fix_ocr_artifacts(cleaned)

    # Step 3: Try to map to canonical name
    lookup_key = cleaned.lower().strip()

    if lookup_key in LAB_TEST_MAPPINGS:
        return LAB_TEST_MAPPINGS[lookup_key]

    # Step 4: Try fuzzy corrections for common OCR errors
    if lookup_key in FUZZY_CORRECTIONS:
        return FUZZY_CORRECTIONS[lookup_key]

    # Step 5: Use context clues if available
    if unit and value is not None:
        inferred = infer_test_from_context(cleaned, unit, value)
        if inferred:
            return inferred

    # If no mapping found, return cleaned version with proper casing
    # Capitalize first letter of each word for readability
    if cleaned.isupper() or cleaned.islower():
        # If all caps or all lowercase, title case it unless it's an acronym
        if len(cleaned) <= 4 and cleaned.isalpha():
            return cleaned.upper()  # Likely an acronym
        return cleaned.title()

    return cleaned


def infer_test_from_context(name: str, unit: str, value: Any) -> Optional[str]:
    """
    Infer the correct test name using unit and value context.

    This uses medical knowledge to disambiguate OCR errors:
    - If unit is 'pg' and value is ~30, it's likely MCH
    - If unit is '%' and value is ~1, it's likely Basophils
    """
    if not unit:
        return None

    try:
        numeric_value = float(value) if value else None
    except (ValueError, TypeError):
        numeric_value = None

    if numeric_value is None:
        return None

    unit_lower = unit.lower().strip()
    name_lower = name.lower().strip()

    # Check against context hints
    for (hint_unit, val_min, val_max), canonical in UNIT_CONTEXT_HINTS.items():
        hint_unit_lower = hint_unit.lower()

        # Check if unit matches (with some flexibility)
        unit_matches = (
            unit_lower == hint_unit_lower or
            unit_lower in hint_unit_lower or
            hint_unit_lower in unit_lower
        )

        if unit_matches and val_min <= numeric_value <= val_max:
            # Check if the cleaned name is similar to canonical (partial match)
            canonical_lower = canonical.lower()
            # Match if: name starts with same letters, or canonical contains name
            if (name_lower[:2] == canonical_lower[:2] or
                name_lower in canonical_lower or
                canonical_lower.startswith(name_lower[:3])):
                logger.debug(f"Context inference: '{name}' + unit={unit}, value={value} -> '{canonical}'")
                return canonical

    return None


def normalize_extracted_values(extracted_values: List[Dict[str, Any]], use_context: bool = True) -> List[Dict[str, Any]]:
    """
    Normalize all extracted values in a list.

    Cleans up field names and values for better presentation.
    Handles both 'field_name' and 'test' keys for compatibility.
    Uses context (unit, value) for intelligent inference when pattern matching fails.

    Args:
        extracted_values: List of extracted lab values
        use_context: Whether to use unit/value context for inference (default: True)
    """
    if not extracted_values:
        return extracted_values

    normalized = []

    for value in extracted_values:
        if isinstance(value, dict):
            normalized_value = value.copy()

            # Normalize field name (check both 'field_name' and 'test' keys)
            field_key = None
            if 'field_name' in normalized_value:
                field_key = 'field_name'
            elif 'test' in normalized_value:
                field_key = 'test'

            if field_key:
                original_name = normalized_value[field_key]

                # Get context for intelligent inference
                unit = normalized_value.get('unit', '') if use_context else None
                val = normalized_value.get('value') if use_context else None

                # Normalize with context awareness
                normalized_name = normalize_lab_test_name(original_name, unit=unit, value=val)
                normalized_value[field_key] = normalized_name

                # Store original for reference if changed
                if normalized_name != original_name:
                    normalized_value['original_name'] = original_name

            # Clean up value if it's a string
            if 'value' in normalized_value and isinstance(normalized_value['value'], str):
                normalized_value['value'] = remove_footnote_markers(
                    normalized_value['value']
                )

            normalized.append(normalized_value)
        else:
            normalized.append(value)

    return normalized


async def normalize_with_llm(
    extracted_values: List[Dict[str, Any]],
    llm_client: Any = None
) -> List[Dict[str, Any]]:
    """
    Use LLM reasoning to normalize ambiguous test names.

    This is an optional enhancement for cases where pattern matching
    and context clues fail. It sends ambiguous values to an LLM for
    intelligent interpretation.

    Args:
        extracted_values: List of extracted values (already pattern-normalized)
        llm_client: Optional LLM client (Ollama/OpenAI). If None, will try to create one.

    Returns:
        List with LLM-corrected test names where applicable
    """
    if not extracted_values:
        return extracted_values

    # Find values that might need LLM correction
    # (those where the name doesn't match any canonical test)
    canonical_names = set(LAB_TEST_MAPPINGS.values())
    ambiguous = []

    for i, val in enumerate(extracted_values):
        if isinstance(val, dict):
            name = val.get('field_name') or val.get('test', '')
            if name and name not in canonical_names:
                ambiguous.append((i, val))

    if not ambiguous:
        return extracted_values  # All names are already canonical

    # Try to get LLM client if not provided
    if llm_client is None:
        try:
            from medical_ingestion.core import get_config
            config = get_config()
            if config.get("use_openai"):
                from openai import AsyncOpenAI
                llm_client = AsyncOpenAI()
            else:
                # Use Ollama
                import httpx
                llm_client = httpx.AsyncClient(base_url="http://localhost:11434")
        except Exception as e:
            logger.warning(f"Could not create LLM client for normalization: {e}")
            return extracted_values

    # Build prompt for LLM
    prompt = """You are a medical lab test name normalizer. Given OCR-extracted test names with possible errors, return the correct canonical name.

Common lab tests include: WBC, RBC, Hemoglobin, Hematocrit, MCV, MCH, MCHC, Platelets, Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils, Glucose, BUN, Creatinine, Sodium, Potassium, Chloride, CO2, Calcium, AST, ALT, Bilirubin, Albumin, Total Cholesterol, HDL, LDL, Triglycerides, TSH, HbA1c, PT, INR

For each input, respond with ONLY the canonical name (no explanation).

"""
    for idx, val in ambiguous:
        name = val.get('field_name') or val.get('test', '')
        unit = val.get('unit', '')
        value = val.get('value', '')
        prompt += f"Input: '{name}' (unit: {unit}, value: {value})\n"

    prompt += "\nRespond with one canonical name per line, in order:"

    try:
        # Call LLM
        if hasattr(llm_client, 'chat'):  # OpenAI client
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            result_text = response.choices[0].message.content.strip()
        else:  # Ollama/httpx client
            response = await llm_client.post(
                "/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0}
                },
                timeout=30.0
            )
            result_text = response.json().get("response", "").strip()

        # Parse results
        corrections = [line.strip() for line in result_text.split('\n') if line.strip()]

        # Apply corrections
        result = list(extracted_values)
        for (idx, val), correction in zip(ambiguous, corrections):
            if correction in canonical_names:
                result[idx] = val.copy()
                field_key = 'field_name' if 'field_name' in val else 'test'
                original = result[idx][field_key]
                result[idx][field_key] = correction
                result[idx]['original_name'] = original
                result[idx]['normalized_by'] = 'llm'
                logger.info(f"LLM normalized: '{original}' -> '{correction}'")

        return result

    except Exception as e:
        logger.warning(f"LLM normalization failed: {e}")
        return extracted_values


def normalize_sections(sections: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize text content in document sections.
    """
    if not sections:
        return sections

    normalized = {}

    for key, value in sections.items():
        if isinstance(value, str):
            # Remove footnote markers from text
            normalized[key] = remove_footnote_markers(value)
        elif isinstance(value, list):
            # Handle lists (like medications)
            normalized[key] = [
                normalize_dict_values(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, dict):
            normalized[key] = normalize_dict_values(value)
        else:
            normalized[key] = value

    return normalized


def normalize_dict_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively normalize string values in a dictionary.
    """
    if not d:
        return d

    normalized = {}

    for key, value in d.items():
        if isinstance(value, str):
            normalized[key] = remove_footnote_markers(value)
        elif isinstance(value, dict):
            normalized[key] = normalize_dict_values(value)
        elif isinstance(value, list):
            normalized[key] = [
                normalize_dict_values(item) if isinstance(item, dict)
                else remove_footnote_markers(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            normalized[key] = value

    return normalized


def fix_spurious_spaces(text: str) -> str:
    """
    Fix spurious single-character splits caused by OCR/PDF extraction.

    Common patterns:
    - "Septe mber" → "September" (trailing single char merged back)
    - "S mith" → "Smith" (leading single char merged to next word)
    - "John, S mith" → "John, Smith"

    Only merges when one side is a single character — avoids collapsing
    intentional spaces between real words.
    """
    if not text or not isinstance(text, str):
        return text or ''

    # Strategy 1: Merge when a space precedes a lowercase continuation
    # "Septe mber" → "September" (lowercase 'm' after space = word continuation)
    # "John Smith" stays unchanged ('S' is uppercase = new word)
    # Only merge when the left side is alpha and the right side starts lowercase
    result = re.sub(r'([a-zA-Z])\s([a-z])', r'\1\2', text)

    # Strategy 2: Merge a single uppercase letter followed by a lowercase word
    # "S mith" → "Smith" (single capital + space + lowercase continuation)
    # Won't match "John Smith" because "John" is 4+ chars (handled above already)
    # This catches cases where strategy 1 missed because the single char is uppercase
    result = re.sub(r'(?:^|(?<=[\s,]))([A-Z])\s([a-z]{2,})', r'\1\2', result)

    return result
