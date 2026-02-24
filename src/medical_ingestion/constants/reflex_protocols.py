# ============================================================================
# src/medical_ingestion/constants/reflex_protocols.py
# ============================================================================
"""
Reflex Testing Protocols
- CAP / AACE / ADA guidelines
- Trigger additional tests based on lab values
"""

import json
from pathlib import Path

# Load from JSON
# Path: constants/ -> medical_ingestion/ -> knowledge/
_knowledge_dir = Path(__file__).parent.parent / "knowledge"

try:
    with open(_knowledge_dir / "reflex_protocols.json") as f:
        REFLEX_PROTOCOLS = json.load(f)
except FileNotFoundError:
    REFLEX_PROTOCOLS = {}
