# ============================================================================
# FILE: src/validators/__init__.py
# ============================================================================
"""
Validators Package

Provides validation for extracted medical data:
- Rule-based validation (fast, deterministic)
- AI validation (context-aware, medical reasoning)
- Conflict resolution (when validators disagree)
- Plausibility checks (catch obvious errors)
"""

from .rule_validator import RuleValidator, validate_value

# AI validator requires transformers - lazy import
try:
    from .ai_validator import AIValidator, validate_with_ai
except ImportError:
    AIValidator = None
    validate_with_ai = None

# Conflict resolver depends on AI validator
try:
    from .conflict_resolver import ConflictResolver, ConflictResolution, resolve_conflict
except ImportError:
    ConflictResolver = None
    ConflictResolution = None
    resolve_conflict = None

from .plausibility import (
    PlausibilityChecker,
    check_plausibility,
    get_plausibility_range
)

__all__ = [
    # Rule validation
    'RuleValidator',
    'validate_value',

    # AI validation
    'AIValidator',
    'validate_with_ai',

    # Conflict resolution
    'ConflictResolver',
    'ConflictResolution',
    'resolve_conflict',

    # Plausibility checks
    'PlausibilityChecker',
    'check_plausibility',
    'get_plausibility_range',
]
