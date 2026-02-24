# ============================================================================
# src/medical_ingestion/core/context/enums.py
# ============================================================================
"""
Processing Enums
- Confidence levels
- Human review priorities
"""

from enum import Enum

class ConfidenceLevel(str, Enum):
    HIGH = "high"       # >= 0.85
    MEDIUM = "medium"   # 0.70 - 0.85
    LOW = "low"         # < 0.70
    CONFLICT = "conflict"  # Validators disagree

class ReviewPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
