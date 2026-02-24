# ============================================================================
# src/medical_ingestion/constants/reference_ranges.py
# ============================================================================
"""
Reference & Plausibility Ranges
- Normal lab ranges (age/sex specific)
- Plausibility ranges to catch extreme errors
"""

import json
from pathlib import Path

# Load from JSON
# Path: constants/ -> medical_ingestion/ -> knowledge/
_knowledge_dir = Path(__file__).parent.parent / "knowledge"

try:
    with open(_knowledge_dir / "reference_ranges.json") as f:
        _ref_data = json.load(f)
except FileNotFoundError:
    _ref_data = {}

try:
    with open(_knowledge_dir / "plausability_ranges.json") as f:
        _plaus_data = json.load(f)
except FileNotFoundError:
    _plaus_data = {}

# Convert to tuple format for backward compatibility
REFERENCE_RANGES = {}
for test_name, ranges in _ref_data.items():
    REFERENCE_RANGES[test_name] = {}
    for category, values in ranges.items():
        REFERENCE_RANGES[test_name][category] = tuple(values)

PLAUSIBILITY_RANGES = {}
for test_name, values in _plaus_data.items():
    PLAUSIBILITY_RANGES[test_name] = tuple(values)
