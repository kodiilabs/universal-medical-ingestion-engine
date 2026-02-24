# ============================================================================
# src/medical_ingestion/constants/loinc.py
# ============================================================================
"""
Common LOINC Codes
- Lab tests: Hematology, Chemistry, Liver, Lipid, Thyroid, Coagulation

Loads from loinc_mappings.json and builds name-to-code lookup dict.
"""

import json
from pathlib import Path

_knowledge_dir = Path(__file__).parent.parent / "knowledge"

# Load LOINC mappings from JSON
with open(_knowledge_dir / "loinc_mappings.json") as f:
    LOINC_MAPPINGS = json.load(f)

# Build name-to-code lookup dict for backward compatibility
LOINC_CODES = {}
for code, data in LOINC_MAPPINGS.items():
    # Add primary name (lowercase, underscored)
    name = data.get("name", "").lower().replace(" ", "_")
    if name:
        LOINC_CODES[name] = code

    # Add aliases
    for alias in data.get("aliases", []):
        alias_key = alias.lower().replace(" ", "_")
        if alias_key not in LOINC_CODES:
            LOINC_CODES[alias_key] = code
