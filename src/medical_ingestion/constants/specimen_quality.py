# ============================================================================
# src/medical_ingestion/constants/specimen_quality.py
# ============================================================================
"""
Specimen Quality Patterns
- Detect hemolysis, lipemia, IV contamination, clotted samples
- Used to flag potentially invalid lab values
"""

import json
from pathlib import Path

# Load from JSON
# Path: constants/ -> medical_ingestion/ -> knowledge/
_knowledge_dir = Path(__file__).parent.parent / "knowledge"

try:
    with open(_knowledge_dir / "specimen_quality_patterns.json") as f:
        SPECIMEN_QUALITY_PATTERNS = json.load(f)
except FileNotFoundError:
    # Fallback to empty dict if file not found
    SPECIMEN_QUALITY_PATTERNS = {}
