# ============================================================================
# src/medical_ingestion/constants/lab_test_db.py
# ============================================================================
"""
Lab Test Database Lookup Utility.

Uses the LOINC SQLite database for comprehensive lab test validation
and LOINC code lookup. Includes fuzzy and OCR-error-tolerant matching.

Mirrors the architecture of medication_db.py for consistency.
"""

import sqlite3
import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any

from .medication_db import levenshtein_distance, soundex

logger = logging.getLogger(__name__)


# Database path - relative to project data directory
# lab_test_db.py -> constants -> medical_ingestion -> src -> project_root
_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "lab_tests"
_KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

# Regional reference range files
# LOINC codes are universal; reference ranges vary by region/population
REGIONAL_REFERENCE_FILES = {
    "usa": "reference_ranges.json",
    "canada": "reference_ranges_ca.json",
    "india": "reference_ranges_in.json",
}

# Current region
_current_region = "usa"

# High-priority clinical name → LOINC code overrides.
# These bypass fuzzy search to fix known mismatches where LOINC's formal
# component naming differs from common clinical usage.
CLINICAL_NAME_OVERRIDES = {
    # Lipid Panel
    "cholesterol total": "2093-3",
    "total cholesterol": "2093-3",
    "cholesterol, total": "2093-3",
    "chol total": "2093-3",
    "chol/hdlc ratio": "9830-1",
    "non-hdl cholesterol": "43396-1",
    "non hdl cholesterol": "43396-1",
    # Hematology — component "Hemoglobin" is shared by HGB, MCH, and MCHC; override to HGB
    "hemoglobin": "718-7",
    "hgb": "718-7",
    "hb": "718-7",
    # CBC — Mean Platelet Volume (LOINC component is "Platelet mean vol", shortname "PMV")
    "mpv": "32623-1",
    "mean platelet volume": "32623-1",
    # Inflammatory markers
    "sed rate": "4537-7",
    "sedimentation rate": "4537-7",
    "esr": "4537-7",
    "erythrocyte sedimentation rate": "4537-7",
    "crp": "1988-5",
    "c-reactive protein": "1988-5",
    "c reactive protein": "1988-5",
    # CBC differentials — absolute counts (LOINC uses "Neutrophils" not "Absolute Neutrophils")
    "absolute neutrophils": "751-8",
    "absolute lymphocytes": "731-0",
    "absolute monocytes": "742-7",
    "absolute eosinophils": "711-2",
    "absolute basophils": "704-7",
    "absolute band neutrophils": "26508-2",
    "immature granulocytes": "38518-7",
    # eGFR
    "egfr non-afr. american": "48642-3",
    "egfr african american": "48643-1",
    "egfr": "48642-3",
}


def set_lab_region(region: str) -> bool:
    """
    Set the lab test reference range region.

    LOINC codes are universal — this only affects reference ranges
    and clinical interpretation thresholds.

    Args:
        region: Region code (usa, canada, india)

    Returns:
        True if region was set, False if invalid
    """
    global _current_region

    region = region.lower().strip()
    if region not in REGIONAL_REFERENCE_FILES:
        logger.error(f"Unknown lab region: '{region}'. Valid regions: {list(REGIONAL_REFERENCE_FILES.keys())}")
        return False

    _current_region = region
    logger.info(f"Lab test region set to: {region}")

    # Reset singleton to reload reference ranges for new region
    LabTestDatabase._instance = None
    return True


def get_lab_region() -> str:
    """Get the current lab test region."""
    return _current_region


def get_db_path() -> Path:
    """Get the lab tests database path."""
    return _DATA_DIR / "lab_tests.db"


class LabTestDatabase:
    """
    SQLite-based lab test lookup for LOINC validation.

    Provides:
    - Exact and fuzzy test name lookup against 96K+ LOINC codes
    - LOINC code retrieval
    - Reference range lookup (region-aware, from JSON)
    - Critical value thresholds
    - OCR-tolerant matching for scanned documents
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._conn = None
        self._available = False
        self._loinc_mappings = {}
        self._connect()
        self._load_loinc_mappings()
        self._initialized = True

    def _connect(self):
        """Establish database connection."""
        db_path = get_db_path()

        if not db_path.exists():
            logger.warning(
                f"Lab tests database not found at {db_path}. "
                "Run 'python data/lab_tests/populate_labtests_db.py' to create it."
            )
            return

        try:
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._available = True
            logger.info(f"Connected to lab tests database: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to lab tests database: {e}")
            self._available = False

    def _load_loinc_mappings(self):
        """Load LOINC mappings JSON for reference ranges and clinical data."""
        try:
            mappings_path = _KNOWLEDGE_DIR / "loinc_mappings.json"
            if mappings_path.exists():
                with open(mappings_path) as f:
                    self._loinc_mappings = json.load(f)
                logger.info(f"Loaded {len(self._loinc_mappings)} LOINC mappings from JSON")
            else:
                logger.warning(f"LOINC mappings not found at {mappings_path}")
        except Exception as e:
            logger.error(f"Failed to load LOINC mappings: {e}")

    @property
    def is_available(self) -> bool:
        """Check if database is available."""
        return self._available

    # =========================================================================
    # LOOKUP METHODS
    # =========================================================================

    def lookup_exact(self, test_name: str) -> Optional[dict]:
        """
        Exact match lookup for a lab test name.

        Searches component, shortname, and consumer_name fields.

        Args:
            test_name: Lab test name to lookup

        Returns:
            Dict with loinc_num, component, shortname, long_common_name, class
            or None if not found
        """
        if not self._available:
            return None

        name_lower = test_name.lower().strip()
        if len(name_lower) < 2:
            return None

        try:
            cursor = self._conn.cursor()

            # Try component, shortname, consumer_name in order
            cursor.execute(
                """
                SELECT loinc_num, component, shortname, long_common_name, class,
                       common_test_rank
                FROM loinc_tests
                WHERE component_lower = ?
                   OR shortname_lower = ?
                   OR consumer_name_lower = ?
                ORDER BY CASE WHEN common_test_rank > 0 THEN common_test_rank ELSE 999999 END ASC
                LIMIT 1
                """,
                (name_lower, name_lower, name_lower)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'loinc_num': row['loinc_num'],
                    'component': row['component'],
                    'shortname': row['shortname'],
                    'long_common_name': row['long_common_name'],
                    'class': row['class'],
                    'match_type': 'exact'
                }
            return None
        except Exception as e:
            logger.error(f"Exact lookup failed: {e}")
            return None

    @lru_cache(maxsize=1000)
    def lookup_fuzzy(self, test_name: str) -> Optional[dict]:
        """
        Fuzzy match lookup with progressive matching strategies.

        Strategy (in order):
        1. Exact match on component/shortname/consumer_name
        2. Whole-word match in relatednames (aliases like "Hb; HGB; Plt")
        3. Shortname prefix match (catches MCV, MCH, MCHC, etc.)
        4. Prefix match on component with length constraint
        5. Word boundary match for multi-word names

        Prefers more common tests (lower common_test_rank).

        Args:
            test_name: Lab test name to lookup

        Returns:
            Dict with loinc_num, component, match_type, etc. or None
        """
        if not self._available:
            return None

        name_lower = test_name.lower().strip()
        if len(name_lower) < 2:
            return None

        try:
            cursor = self._conn.cursor()

            # 0. Clinical name overrides (highest priority — fixes known mismatches)
            override_loinc = CLINICAL_NAME_OVERRIDES.get(name_lower)
            if override_loinc:
                cursor.execute(
                    "SELECT loinc_num, component, shortname, long_common_name, class "
                    "FROM loinc_tests WHERE loinc_num = ?",
                    (override_loinc,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'loinc_num': row['loinc_num'],
                        'component': row['component'],
                        'shortname': row['shortname'],
                        'long_common_name': row['long_common_name'],
                        'class': row['class'],
                        'match_type': 'clinical_override'
                    }

            # 1. Exact match
            result = self.lookup_exact(test_name)
            if result:
                return result

            # 2. Combined alias + shortname scoring
            # Fetch candidates from both relatednames and shortname prefix,
            # then score to pick the best match. This handles abbreviations
            # that appear in multiple tests (e.g., "Hb" in Hemoglobin and HbA1c).
            candidates = []
            seen_loinc = set()

            # 2a. Candidates from relatednames (prioritize common tests)
            cursor.execute(
                """
                SELECT loinc_num, component, shortname, long_common_name, class,
                       common_test_rank, relatednames, shortname_lower, component_lower
                FROM loinc_tests
                WHERE relatednames LIKE ?
                ORDER BY CASE WHEN common_test_rank > 0 THEN common_test_rank ELSE 999999 END ASC
                LIMIT 50
                """,
                ('%' + name_lower + '%',)
            )
            for row in cursor.fetchall():
                if row['loinc_num'] not in seen_loinc:
                    seen_loinc.add(row['loinc_num'])
                    candidates.append(row)

            # 2b. Candidates from shortname prefix
            cursor.execute(
                """
                SELECT loinc_num, component, shortname, long_common_name, class,
                       common_test_rank, relatednames, shortname_lower, component_lower
                FROM loinc_tests
                WHERE shortname_lower LIKE ?
                ORDER BY CASE WHEN common_test_rank > 0 THEN common_test_rank ELSE 999999 END ASC
                LIMIT 20
                """,
                (name_lower + ' %',)
            )
            for row in cursor.fetchall():
                if row['loinc_num'] not in seen_loinc:
                    seen_loinc.add(row['loinc_num'])
                    candidates.append(row)

            # Score candidates
            scored = []
            for row in candidates:
                related = (row['relatednames'] or '').lower()
                aliases = [a.strip() for a in related.split(';')]
                shortname_l = (row['shortname_lower'] or '')
                component_l = (row['component_lower'] or '')

                is_alias = name_lower in aliases
                # Word-boundary shortname match (avoid "hb" matching "hba1c")
                shortname_match = (
                    shortname_l == name_lower
                    or shortname_l.startswith(name_lower + ' ')
                    or shortname_l.startswith(name_lower + '-')
                )

                if not is_alias and not shortname_match:
                    continue

                score = 0
                if shortname_match:
                    score += 200  # Strong: shortname starts with search term at word boundary
                if is_alias:
                    score += 100  # Confirmed alias in relatednames
                    # Alias position bonus: earlier = more central to the test
                    # "Hb" at position 2 in Hemoglobin vs position 7 in MCHC
                    try:
                        pos = aliases.index(name_lower)
                        score += max(0, 50 - pos * 5)  # Up to +50 for first alias
                    except ValueError:
                        pass
                if component_l.startswith(name_lower):
                    score += 50
                rank = row['common_test_rank'] or 999999
                if 0 < rank < 999999:
                    score += (1000 - min(rank, 1000))
                scored.append((score, row))

            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                best = scored[0][1]
                match_type = 'alias'
                # Determine match type based on what matched
                shortname_l = (best['shortname_lower'] or '')
                if (shortname_l == name_lower or shortname_l.startswith(name_lower + ' ')
                        or shortname_l.startswith(name_lower + '-')):
                    match_type = 'shortname_prefix'
                return {
                    'loinc_num': best['loinc_num'],
                    'component': best['component'],
                    'shortname': best['shortname'],
                    'long_common_name': best['long_common_name'],
                    'class': best['class'],
                    'match_type': match_type
                }

            # 4. Prefix match on component (with length constraint to avoid false positives)
            if len(name_lower) >= 3:
                cursor.execute(
                    """
                    SELECT loinc_num, component, shortname, long_common_name, class,
                           common_test_rank
                    FROM loinc_tests
                    WHERE component_lower LIKE ?
                      AND LENGTH(component_lower) <= ?
                    ORDER BY CASE WHEN common_test_rank > 0 THEN common_test_rank ELSE 999999 END ASC
                    LIMIT 1
                    """,
                    (name_lower + '%', len(name_lower) + 3)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'loinc_num': row['loinc_num'],
                        'component': row['component'],
                        'shortname': row['shortname'],
                        'long_common_name': row['long_common_name'],
                        'class': row['class'],
                        'match_type': 'prefix'
                    }

            # 5. Word boundary match (e.g., "glucose" matches "Glucose [Mass/volume]...")
            if len(name_lower) >= 4:
                cursor.execute(
                    """
                    SELECT loinc_num, component, shortname, long_common_name, class,
                           common_test_rank
                    FROM loinc_tests
                    WHERE component_lower LIKE ?
                    ORDER BY CASE WHEN common_test_rank > 0 THEN common_test_rank ELSE 999999 END ASC
                    LIMIT 1
                    """,
                    (name_lower + ' %',)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'loinc_num': row['loinc_num'],
                        'component': row['component'],
                        'shortname': row['shortname'],
                        'long_common_name': row['long_common_name'],
                        'class': row['class'],
                        'match_type': 'word_match'
                    }

            return None
        except Exception as e:
            logger.error(f"Fuzzy lookup failed: {e}")
            return None

    def lookup_ocr_tolerant(
        self,
        test_name: str,
        max_distance: int = None,
        expected_unit: str = None
    ) -> Optional[dict]:
        """
        OCR-error-tolerant lab test lookup using edit distance.

        Less aggressive than medication OCR correction since lab reports
        are usually printed. Handles:
        - Minor typos and OCR character substitutions
        - Abbreviated names (HGB, BUN, etc.)

        Args:
            test_name: Potentially OCR-corrupted test name
            max_distance: Maximum Levenshtein distance (default: dynamic)
            expected_unit: Expected unit for cross-validation

        Returns:
            Dict with loinc_num, match_type='ocr_corrected', similarity, etc.
        """
        if not self._available:
            return None

        name_lower = test_name.lower().strip()
        if len(name_lower) < 3:
            return None

        # Conservative max distance for lab tests (less OCR noise than prescriptions)
        if max_distance is None:
            if len(name_lower) <= 4:
                max_distance = 1
            elif len(name_lower) <= 6:
                max_distance = 2
            else:
                max_distance = 3

        try:
            cursor = self._conn.cursor()

            # Get candidates: same first letter, similar length, common tests preferred
            min_len = max(2, len(name_lower) - max_distance)
            max_len = len(name_lower) + max_distance + 2

            cursor.execute(
                """
                SELECT DISTINCT loinc_num, component, shortname, long_common_name,
                       class, common_test_rank, component_lower
                FROM loinc_tests
                WHERE component_lower LIKE ?
                  AND LENGTH(component_lower) BETWEEN ? AND ?
                ORDER BY CASE WHEN common_test_rank > 0 THEN common_test_rank ELSE 999999 END ASC
                LIMIT 500
                """,
                (name_lower[0] + '%', min_len, max_len)
            )
            candidates = list(cursor.fetchall())

            # Also check shortnames with same first letter
            cursor.execute(
                """
                SELECT DISTINCT loinc_num, component, shortname, long_common_name,
                       class, common_test_rank, shortname_lower as component_lower
                FROM loinc_tests
                WHERE shortname_lower LIKE ?
                  AND LENGTH(shortname_lower) BETWEEN ? AND ?
                ORDER BY CASE WHEN common_test_rank > 0 THEN common_test_rank ELSE 999999 END ASC
                LIMIT 200
                """,
                (name_lower[0] + '%', min_len, max_len)
            )
            candidates.extend(cursor.fetchall())

            # Score candidates
            best_match = None
            best_score = -1
            best_distance = max_distance + 1
            input_soundex = soundex(name_lower)

            seen_loinc = set()
            for row in candidates:
                loinc = row['loinc_num']
                if loinc in seen_loinc:
                    continue
                seen_loinc.add(loinc)

                candidate_name = row['component_lower']
                if not candidate_name:
                    continue

                distance = levenshtein_distance(name_lower, candidate_name)
                if distance > max_distance:
                    continue

                max_len_pair = max(len(name_lower), len(candidate_name))
                similarity = 1 - (distance / max_len_pair)

                # Popularity bonus (lower rank = more common)
                rank = row['common_test_rank'] or 999999
                rank_bonus = 0.2 if rank <= 100 else (0.1 if rank <= 500 else 0.0)

                # Phonetic bonus
                phonetic_bonus = 0.1 if soundex(candidate_name) == input_soundex else 0.0

                # Unit class bonus
                unit_bonus = 0.0
                if expected_unit and row['class']:
                    test_class = row['class'].upper()
                    unit_upper = expected_unit.upper()
                    # Basic heuristic: chemistry units match chemistry tests
                    if ('CHEM' in test_class and unit_upper in ('MG/DL', 'MMOL/L', 'MEQ/L')):
                        unit_bonus = 0.1
                    elif ('HEM' in test_class and unit_upper in ('G/DL', 'K/UL', 'M/UL', '%')):
                        unit_bonus = 0.1

                score = similarity + rank_bonus + phonetic_bonus + unit_bonus

                if score > best_score:
                    best_score = score
                    best_match = row
                    best_distance = distance

            if best_match:
                return {
                    'loinc_num': best_match['loinc_num'],
                    'component': best_match['component'],
                    'shortname': best_match['shortname'],
                    'long_common_name': best_match['long_common_name'],
                    'class': best_match['class'],
                    'match_type': 'ocr_corrected',
                    'edit_distance': best_distance,
                    'similarity': round(1 - (best_distance / max(len(name_lower), len(best_match['component_lower'] or ''), 1)), 2),
                    'original_input': test_name,
                    'score': round(best_score, 3)
                }

            return None
        except Exception as e:
            logger.error(f"OCR-tolerant lookup failed: {e}")
            return None

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def is_valid_test(self, test_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a string is a valid lab test name in LOINC.

        Args:
            test_name: Name to validate

        Returns:
            Tuple of (is_valid, loinc_code)
        """
        result = self.lookup_fuzzy(test_name)
        if result:
            return True, result.get('loinc_num')
        return False, None

    def get_loinc_code(self, test_name: str) -> Optional[str]:
        """
        Get LOINC code for a lab test name.

        Args:
            test_name: Lab test name

        Returns:
            LOINC code or None
        """
        result = self.lookup_fuzzy(test_name)
        return result.get('loinc_num') if result else None

    def get_reference_range(
        self,
        loinc_code: str,
        patient_sex: str = None,
        patient_age: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get reference range for a LOINC code from the knowledge base.

        Uses loinc_mappings.json (not the SQLite database) since
        reference ranges are clinical data, not LOINC coding data.

        Args:
            loinc_code: LOINC code (e.g., "718-7")
            patient_sex: "male" or "female" (for sex-specific ranges)
            patient_age: Patient age in years

        Returns:
            Dict with low, high, unit or None
        """
        mapping = self._loinc_mappings.get(loinc_code)
        if not mapping or 'reference_range' not in mapping:
            return None

        ref_ranges = mapping['reference_range']

        # Try sex-specific range first
        if patient_sex:
            sex_key = patient_sex.lower().strip()
            if sex_key in ref_ranges:
                return ref_ranges[sex_key]

        # Fall back to adult/general range
        if 'adult' in ref_ranges:
            return ref_ranges['adult']

        # Return first available range
        if ref_ranges:
            first_key = next(iter(ref_ranges))
            return ref_ranges[first_key]

        return None

    def get_expected_unit(self, loinc_code: str) -> Optional[str]:
        """
        Get the standard unit for a LOINC code.

        Args:
            loinc_code: LOINC code

        Returns:
            Standard unit string (e.g., "g/dL") or None
        """
        mapping = self._loinc_mappings.get(loinc_code)
        if mapping:
            return mapping.get('standard_unit')
        return None

    def get_critical_values(self, loinc_code: str) -> Optional[Dict[str, Any]]:
        """
        Get critical value thresholds for a LOINC code.

        Critical values are extreme results that require immediate clinical action.

        Args:
            loinc_code: LOINC code

        Returns:
            Dict with low, high, unit or None
        """
        mapping = self._loinc_mappings.get(loinc_code)
        if mapping:
            return mapping.get('critical_values')
        return None

    def search(self, query: str, limit: int = 10) -> List[dict]:
        """
        Free-text search across lab test names.

        Searches component, shortname, and relatednames fields.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching test dicts
        """
        if not self._available or len(query) < 2:
            return []

        try:
            cursor = self._conn.cursor()
            query_lower = query.lower()

            cursor.execute(
                """
                SELECT DISTINCT loinc_num, component, shortname, long_common_name,
                       class, common_test_rank
                FROM loinc_tests
                WHERE component_lower LIKE ?
                   OR shortname_lower LIKE ?
                   OR consumer_name_lower LIKE ?
                   OR relatednames LIKE ?
                ORDER BY
                    CASE WHEN component_lower = ? THEN 0
                         WHEN component_lower LIKE ? THEN 1
                         ELSE 2 END,
                    common_test_rank ASC
                LIMIT ?
                """,
                (
                    '%' + query_lower + '%',
                    '%' + query_lower + '%',
                    '%' + query_lower + '%',
                    '%' + query_lower + '%',
                    query_lower,
                    query_lower + '%',
                    limit
                )
            )
            return [
                {
                    'loinc_num': row['loinc_num'],
                    'component': row['component'],
                    'shortname': row['shortname'],
                    'long_common_name': row['long_common_name'],
                    'class': row['class']
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._available = False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_db_instance: Optional[LabTestDatabase] = None


def get_lab_test_db() -> LabTestDatabase:
    """Get the singleton lab test database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = LabTestDatabase()
    return _db_instance


def is_valid_test(test_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a string is a valid lab test name.

    Args:
        test_name: Name to validate

    Returns:
        Tuple of (is_valid, loinc_code)
    """
    db = get_lab_test_db()
    return db.is_valid_test(test_name)


def get_loinc_code(test_name: str) -> Optional[str]:
    """
    Get LOINC code for a lab test name.

    Args:
        test_name: Lab test name

    Returns:
        LOINC code or None
    """
    db = get_lab_test_db()
    return db.get_loinc_code(test_name)


def lookup_lab_test(
    test_name: str,
    ocr_correction: bool = True,
    expected_unit: str = None
) -> Optional[dict]:
    """
    Look up lab test info with optional OCR error correction.

    Tries exact/fuzzy match first, then falls back to OCR-tolerant
    matching if no direct match is found.

    Args:
        test_name: Lab test name to lookup
        ocr_correction: Whether to try OCR error correction (default True)
        expected_unit: Expected unit for cross-validation

    Returns:
        Dict with loinc_num, component, match_type, etc. or None
    """
    db = get_lab_test_db()

    # Try exact/fuzzy match first
    result = db.lookup_fuzzy(test_name)
    if result:
        return result

    # If no match and OCR correction enabled, try OCR-tolerant lookup
    if ocr_correction:
        result = db.lookup_ocr_tolerant(
            test_name,
            expected_unit=expected_unit
        )
        if result:
            logger.info(
                f"OCR correction: '{test_name}' -> '{result['component']}' "
                f"(distance: {result['edit_distance']}, similarity: {result['similarity']})"
            )
            return result

    return None
