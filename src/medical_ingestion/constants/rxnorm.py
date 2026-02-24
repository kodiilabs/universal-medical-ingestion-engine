# ============================================================================
# src/medical_ingestion/constants/rxnorm.py
# ============================================================================
"""
RxNorm medication codes and mappings.

Loads from knowledge base JSON file.
"""

import json
from pathlib import Path

_knowledge_dir = Path(__file__).parent.parent / "knowledge"

with open(_knowledge_dir / "rxnorm_mappings.json") as f:
    RXNORM_MAPPINGS = json.load(f)

# Build lookup dictionaries
RXNORM_CODES = {}
RXNORM_BY_GENERIC = {}
RXNORM_BY_BRAND = {}
RXNORM_BY_CLASS = {}

for code, data in RXNORM_MAPPINGS.items():
    generic_name = data.get("generic_name", "")

    # Map generic name to code
    if generic_name:
        RXNORM_CODES[generic_name.lower()] = code
        RXNORM_BY_GENERIC[generic_name.lower()] = data

    # Map brand names to code
    for brand in data.get("brand_names", []):
        brand_lower = brand.lower()
        RXNORM_CODES[brand_lower] = code
        RXNORM_BY_BRAND[brand_lower] = data

    # Group by drug class
    drug_class = data.get("class", "")
    if drug_class:
        if drug_class not in RXNORM_BY_CLASS:
            RXNORM_BY_CLASS[drug_class] = []
        RXNORM_BY_CLASS[drug_class].append({
            "code": code,
            "generic_name": generic_name,
            **data
        })


def get_rxnorm_code(medication_name: str) -> str | None:
    """
    Look up RxNorm code for a medication name.

    Args:
        medication_name: Generic or brand name

    Returns:
        RxNorm code or None if not found
    """
    return RXNORM_CODES.get(medication_name.lower())


def get_medication_info(medication_name: str) -> dict | None:
    """
    Get full medication info for a name.

    Args:
        medication_name: Generic or brand name

    Returns:
        Medication data dict or None if not found
    """
    name_lower = medication_name.lower()

    if name_lower in RXNORM_BY_GENERIC:
        return RXNORM_BY_GENERIC[name_lower]

    if name_lower in RXNORM_BY_BRAND:
        return RXNORM_BY_BRAND[name_lower]

    return None


def get_medications_by_class(drug_class: str) -> list:
    """
    Get all medications in a drug class.

    Args:
        drug_class: Drug class name (e.g., "statin", "SSRI")

    Returns:
        List of medication data dicts
    """
    return RXNORM_BY_CLASS.get(drug_class, [])
