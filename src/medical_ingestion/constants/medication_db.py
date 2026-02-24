# ============================================================================
# src/medical_ingestion/constants/medication_db.py
# ============================================================================
"""
Medication Database Lookup Utility.

Uses the RxNorm SQLite database for comprehensive medication validation
and RxCUI code lookup. Includes OCR-error-tolerant fuzzy matching.
"""

import sqlite3
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def soundex(name: str) -> str:
    """
    Generate Soundex code for phonetic matching.
    Helps match medications that sound similar despite OCR errors.
    """
    name = name.upper()
    if not name:
        return ""

    # Keep first letter
    soundex_code = name[0]

    # Soundex mapping
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
        'A': '0', 'E': '0', 'I': '0', 'O': '0', 'U': '0', 'H': '0', 'W': '0', 'Y': '0'
    }

    # Encode remaining characters
    prev_code = mapping.get(name[0], '0')
    for char in name[1:]:
        code = mapping.get(char, '0')
        if code != '0' and code != prev_code:
            soundex_code += code
        prev_code = code

    # Pad or truncate to 4 characters
    soundex_code = (soundex_code + '000')[:4]
    return soundex_code


# Common OCR character confusions in handwritten prescriptions
OCR_CONFUSIONS = {
    # Character -> list of commonly confused characters
    't': ['l', 'i', 'f', 'k'],      # t can look like k in handwriting
    'l': ['t', 'i', '1'],
    'i': ['l', 't', '1', 'j'],
    'n': ['u', 'r', 'h', 'm'],
    'u': ['n', 'v', 'a'],
    'm': ['rn', 'nn', 'n'],
    'rn': ['m'],
    'a': ['o', 'e', 'u'],
    'o': ['a', 'e', '0', 'c', 'd'],  # o can look like d in handwriting
    'e': ['o', 'a', 'c'],
    'c': ['o', 'e'],
    'h': ['n', 'b', 'k'],
    'r': ['n', 'v'],
    'v': ['u', 'r', 'w'],
    'w': ['vv', 'uu'],
    's': ['5', 'z'],
    'z': ['2', 's'],
    'g': ['q', '9'],
    'q': ['g', '9'],
    'd': ['cl', 'b', 'o', 'a'],     # d can look like o or a in handwriting
    'b': ['d', 'h', '6'],
    'p': ['b', 'q'],
    'f': ['t', 'l'],
    'k': ['t', 'h', 'lc', 'x'],     # k can look like t in handwriting (added!)
    'dd': ['ol', 'al', 'cl'],       # double d can be misread as ol/al (added!)
}


def generate_ocr_variants(name: str, max_variants: int = 20) -> List[str]:
    """
    Generate possible OCR error variants of a medication name.

    Creates variants by applying common character confusions.
    This helps match when OCR misreads specific characters.

    Args:
        name: Original medication name
        max_variants: Maximum variants to generate

    Returns:
        List of possible correct spellings
    """
    variants = set()
    name_lower = name.lower()

    # Single character substitutions
    for i, char in enumerate(name_lower):
        if char in OCR_CONFUSIONS:
            for replacement in OCR_CONFUSIONS[char]:
                variant = name_lower[:i] + replacement + name_lower[i+1:]
                variants.add(variant)

    # Two-character sequence substitutions (rn -> m, etc.)
    for i in range(len(name_lower) - 1):
        two_char = name_lower[i:i+2]
        if two_char in OCR_CONFUSIONS:
            for replacement in OCR_CONFUSIONS[two_char]:
                variant = name_lower[:i] + replacement + name_lower[i+2:]
                variants.add(variant)

    # Also try expanding single chars to two-char sequences (m -> rn)
    for i, char in enumerate(name_lower):
        if char == 'm':
            variants.add(name_lower[:i] + 'rn' + name_lower[i+1:])
        elif char == 'w':
            variants.add(name_lower[:i] + 'vv' + name_lower[i+1:])

    return list(variants)[:max_variants]


# Drug class patterns for context scoring
DRUG_CLASS_PATTERNS = {
    'beta_blocker': {
        # Include 'ol' to catch OCR errors that truncate 'olol' to 'ol'
        'suffixes': ['olol', 'alol', 'ol'],
        'examples': ['atenolol', 'metoprolol', 'propranolol', 'carvedilol', 'bisoprolol'],
    },
    'ace_inhibitor': {
        'suffixes': ['pril'],
        'examples': ['lisinopril', 'enalapril', 'ramipril', 'benazepril', 'captopril'],
    },
    'arb': {
        'suffixes': ['sartan'],
        'examples': ['losartan', 'valsartan', 'irbesartan', 'olmesartan', 'candesartan'],
    },
    'statin': {
        'suffixes': ['statin'],
        'examples': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin'],
    },
    'ppi': {
        'suffixes': ['prazole'],
        'examples': ['omeprazole', 'pantoprazole', 'esomeprazole', 'lansoprazole'],
    },
    'ssri': {
        'suffixes': [],
        'examples': ['sertraline', 'fluoxetine', 'escitalopram', 'citalopram', 'paroxetine'],
    },
    'antibiotic': {
        'suffixes': ['cillin', 'mycin', 'cycline', 'floxacin'],
        'examples': ['amoxicillin', 'azithromycin', 'doxycycline', 'ciprofloxacin'],
    },
    'benzo': {
        'suffixes': ['pam', 'lam'],
        'examples': ['lorazepam', 'alprazolam', 'diazepam', 'clonazepam'],
    },
    'diuretic': {
        'suffixes': ['thiazide', 'semide'],
        'examples': ['hydrochlorothiazide', 'furosemide', 'spironolactone'],
    },
    'calcium_blocker': {
        'suffixes': ['dipine'],
        'examples': ['amlodipine', 'nifedipine', 'diltiazem', 'verapamil'],
    },
    'diabetes': {
        'suffixes': ['gliflozin', 'glutide', 'gliptin'],
        'examples': ['metformin', 'glipizide', 'glyburide', 'empagliflozin', 'semaglutide'],
    },
    'anticonvulsant': {
        'suffixes': [],
        'examples': ['gabapentin', 'pregabalin', 'levetiracetam', 'lamotrigine', 'topiramate'],
    },
    'opioid': {
        'suffixes': ['codone', 'morphone'],
        'examples': ['hydrocodone', 'oxycodone', 'tramadol', 'morphine', 'fentanyl'],
    },
}


def infer_drug_class(medication_name: str) -> Optional[str]:
    """
    Infer the drug class of a medication based on name patterns.

    Args:
        medication_name: Medication name

    Returns:
        Drug class name or None
    """
    name_lower = medication_name.lower()

    for drug_class, patterns in DRUG_CLASS_PATTERNS.items():
        # Check suffixes
        for suffix in patterns['suffixes']:
            if name_lower.endswith(suffix):
                return drug_class

        # Check exact matches
        for example in patterns['examples']:
            if example in name_lower or name_lower in example:
                return drug_class

    return None

# Database path - relative to project data directory
# medication_db.py -> constants -> medical_ingestion -> src -> project_root
_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "medications"

# Regional database paths - add new regions here as databases become available
# The key is the region code, value is the database filename
REGIONAL_DATABASES = {
    "usa": "medications.db",         # US RxNorm database (default)
    "canada": "medications_ca.db",   # Canadian Drug Product Database (future)
    "india": "medications_in.db",    # Indian drug database (future)
}

# Current region - can be set via set_medication_region()
_current_region = "usa"


def set_medication_region(region: str) -> bool:
    """
    Set the medication database region.

    Args:
        region: Region code (usa, canada, india)

    Returns:
        True if region was set, False if invalid region
    """
    global _current_region

    region = region.lower().strip()
    if region not in REGIONAL_DATABASES:
        logger.error(f"Unknown medication region: '{region}'. Valid regions: {list(REGIONAL_DATABASES.keys())}")
        return False

    _current_region = region
    logger.info(f"Medication region set to: {region}")

    # Reset the singleton to reload with new region's database
    MedicationDatabase._instance = None
    return True


def get_medication_region() -> str:
    """Get the current medication region."""
    return _current_region


def get_db_path() -> Path:
    """Get the database path for the current region."""
    db_file = REGIONAL_DATABASES.get(_current_region, "medications.db")
    return _DATA_DIR / db_file


class MedicationDatabase:
    """
    SQLite-based medication lookup for RxNorm validation.

    Provides:
    - Exact and fuzzy medication name lookup
    - RxCUI code retrieval
    - Medication validation
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
        self._connect()
        self._initialized = True

    def _connect(self):
        """Establish database connection for the current region."""
        db_path = get_db_path()
        self._region = get_medication_region()

        if not db_path.exists():
            logger.warning(
                f"Medication database not found at {db_path} for region '{self._region}'. "
                "Run 'python data/medications/populate_medicationdb.py' to create it."
            )
            return

        try:
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._available = True
            logger.info(f"Connected to medication database: {db_path} (region: {self._region})")
        except Exception as e:
            logger.error(f"Failed to connect to medication database: {e}")
            self._available = False

    @property
    def region(self) -> str:
        """Get the database region."""
        return getattr(self, '_region', 'usa')

    @property
    def is_available(self) -> bool:
        """Check if database is available."""
        return self._available

    def lookup_exact(self, medication_name: str) -> Optional[dict]:
        """
        Exact match lookup for medication name.

        Args:
            medication_name: Medication name to lookup

        Returns:
            Dict with rxcui, name, term_type or None if not found
        """
        if not self._available:
            return None

        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT rxcui, name, term_type
                FROM medications
                WHERE name_lower = ?
                LIMIT 1
                """,
                (medication_name.lower().strip(),)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'rxcui': row['rxcui'],
                    'name': row['name'],
                    'term_type': row['term_type']
                }
            return None
        except Exception as e:
            logger.error(f"Database lookup failed: {e}")
            return None

    @lru_cache(maxsize=1000)
    def lookup_fuzzy(self, medication_name: str) -> Optional[dict]:
        """
        Fuzzy match lookup - finds medications with conservative matching.

        Matching strategy (in order):
        1. Exact match (highest confidence)
        2. Case-insensitive exact match
        3. Prefix match only if result length is similar (avoid Aerol -> AeroLEF)
        4. Word boundary match for multi-word names

        Args:
            medication_name: Medication name to lookup

        Returns:
            Dict with rxcui, name, term_type, match_type or None if not found
        """
        if not self._available:
            return None

        name_lower = medication_name.lower().strip()

        # Skip very short names to avoid false positives
        if len(name_lower) < 3:
            return None

        try:
            cursor = self._conn.cursor()

            # 1. Try exact match first
            cursor.execute(
                "SELECT rxcui, name, term_type FROM medications WHERE name_lower = ? LIMIT 1",
                (name_lower,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'rxcui': row['rxcui'],
                    'name': row['name'],
                    'term_type': row['term_type'],
                    'match_type': 'exact'
                }

            # 2. Try prefix match but only for very close matches
            # This prevents "Aerol" matching "AeroLEF" (different medication)
            # Only match if the database name is at most 1 char longer (e.g., trailing 's')
            cursor.execute(
                """
                SELECT rxcui, name, term_type FROM medications
                WHERE name_lower LIKE ?
                AND LENGTH(name_lower) <= ?
                ORDER BY LENGTH(name) ASC
                LIMIT 1
                """,
                (name_lower + '%', len(name_lower) + 1)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'rxcui': row['rxcui'],
                    'name': row['name'],
                    'term_type': row['term_type'],
                    'match_type': 'prefix'
                }

            # 3. Try matching as first word in multi-word medication names
            # e.g., "metformin" matches "metformin hydrochloride"
            if len(name_lower) >= 4:
                cursor.execute(
                    """
                    SELECT rxcui, name, term_type FROM medications
                    WHERE name_lower LIKE ?
                    AND term_type IN ('IN', 'BN', 'PIN', 'MIN')
                    ORDER BY LENGTH(name) ASC
                    LIMIT 1
                    """,
                    (name_lower + ' %',)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'rxcui': row['rxcui'],
                        'name': row['name'],
                        'term_type': row['term_type'],
                        'match_type': 'word_prefix'
                    }

            return None
        except Exception as e:
            logger.error(f"Fuzzy lookup failed: {e}")
            return None

    def is_valid_medication(self, medication_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a string is a valid medication name in RxNorm.

        Args:
            medication_name: Name to validate

        Returns:
            Tuple of (is_valid, rxcui)
        """
        result = self.lookup_fuzzy(medication_name)
        if result:
            return True, result.get('rxcui')
        return False, None

    def get_rxcui(self, medication_name: str) -> Optional[str]:
        """
        Get RxCUI code for a medication name.

        Args:
            medication_name: Medication name

        Returns:
            RxCUI code or None
        """
        result = self.lookup_fuzzy(medication_name)
        return result.get('rxcui') if result else None

    def search(self, query: str, limit: int = 10) -> List[dict]:
        """
        Search for medications matching a query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching medication dicts
        """
        if not self._available or len(query) < 2:
            return []

        try:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT rxcui, name, term_type
                FROM medications
                WHERE name_lower LIKE ?
                ORDER BY
                    CASE WHEN name_lower = ? THEN 0
                         WHEN name_lower LIKE ? THEN 1
                         ELSE 2 END,
                    LENGTH(name)
                LIMIT ?
                """,
                (
                    '%' + query.lower() + '%',
                    query.lower(),
                    query.lower() + '%',
                    limit
                )
            )
            return [
                {
                    'rxcui': row['rxcui'],
                    'name': row['name'],
                    'term_type': row['term_type']
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_medication_strengths(self, rxcui: str) -> List[str]:
        """
        Get all known strengths for a medication by RxCUI.

        First tries medication_dosages table, then falls back to extracting
        strengths from related SCD/SBD medication names.

        Args:
            rxcui: RxNorm Concept Unique Identifier

        Returns:
            List of strength strings (e.g., ['25 MG', '50 MG', '100 MG'])
        """
        if not self._available:
            return []

        try:
            import re
            cursor = self._conn.cursor()

            # Try medication_dosages table first
            cursor.execute(
                """
                SELECT DISTINCT strength
                FROM medication_dosages
                WHERE rxcui = ?
                AND strength IS NOT NULL
                ORDER BY CAST(REPLACE(REPLACE(strength, ' MG', ''), ' MCG', '') AS REAL)
                """,
                (rxcui,)
            )
            strengths = [row['strength'] for row in cursor.fetchall()]

            if strengths:
                return strengths

            # Fallback: Get the medication name and look for related SCD/SBD entries
            cursor.execute(
                "SELECT name_lower FROM medications WHERE rxcui = ? LIMIT 1",
                (rxcui,)
            )
            row = cursor.fetchone()
            if not row:
                return []

            base_name = row['name_lower'].split()[0]  # Get first word (e.g., "atenolol")

            # Find SCD entries with this ingredient and extract strengths from their names
            cursor.execute(
                """
                SELECT DISTINCT name FROM medications
                WHERE name_lower LIKE ?
                AND term_type IN ('SCD', 'SBD')
                LIMIT 50
                """,
                (base_name + ' %',)
            )

            # Extract strengths from names like "atenolol 25 MG Oral Tablet"
            strength_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(MG|MCG|G|ML|%)', re.IGNORECASE)
            extracted_strengths = set()

            for row in cursor.fetchall():
                match = strength_pattern.search(row['name'])
                if match:
                    strength = f"{match.group(1)} {match.group(2).upper()}"
                    extracted_strengths.add(strength)

            # Sort by numeric value
            def sort_key(s):
                try:
                    return float(re.search(r'([\d.]+)', s).group(1))
                except:
                    return 0

            return sorted(list(extracted_strengths), key=sort_key)

        except Exception as e:
            logger.debug(f"Strength lookup failed: {e}")
            return []

    def validate_strength(self, rxcui: str, extracted_strength: str) -> Tuple[bool, float]:
        """
        Validate if an extracted strength matches known strengths for a medication.

        Args:
            rxcui: RxNorm Concept Unique Identifier
            extracted_strength: Strength extracted from prescription (e.g., "25mg", "25 mg", "Z5my")

        Returns:
            Tuple of (is_valid, confidence_boost)
            - is_valid: True if strength matches a known strength
            - confidence_boost: Additional confidence to add (0.0-0.2)
        """
        if not extracted_strength or not rxcui:
            return False, 0.0

        known_strengths = self.get_medication_strengths(rxcui)
        if not known_strengths:
            return False, 0.0

        # Normalize extracted strength for comparison
        import re

        # First, fix common OCR errors in the strength string
        # Z->2, O->0, l->1, S->5, B->8, I->1, etc.
        ocr_fixed = extracted_strength.upper().strip()
        ocr_substitutions = [
            ('Z', '2'),   # Z often OCR'd for 2
            ('O', '0'),   # O often OCR'd for 0
            ('L', '1'),   # lowercase l often OCR'd for 1
            ('I', '1'),   # I often OCR'd for 1
            ('S', '5'),   # S sometimes OCR'd for 5
            ('B', '8'),   # B sometimes OCR'd for 8
            ('MY', 'MG'), # my is OCR error for mg
        ]

        # Apply substitutions only to the numeric portion (before unit)
        # Split into numeric and unit parts
        match = re.match(r'^([A-Z\d.]+)\s*(MG|MY|MCG|G|ML|%|UNIT|UNITS)?(.*)$', ocr_fixed, re.IGNORECASE)
        if match:
            numeric_part = match.group(1)
            unit_part = match.group(2) or ''
            rest = match.group(3) or ''

            # Apply OCR fixes to numeric part
            for ocr_char, digit in ocr_substitutions[:5]:  # First 5 are digit substitutions
                numeric_part = numeric_part.replace(ocr_char, digit)

            # Fix unit (MY -> MG)
            if unit_part.upper() == 'MY':
                unit_part = 'MG'

            ocr_fixed = numeric_part + ' ' + unit_part + rest
        else:
            # Fallback: just apply all substitutions
            for ocr_char, replacement in ocr_substitutions:
                ocr_fixed = ocr_fixed.replace(ocr_char, replacement)

        extracted_normalized = re.sub(r'\s+', ' ', ocr_fixed.strip())
        # Handle "25MG" -> "25 MG"
        extracted_normalized = re.sub(r'(\d+)(MG|MCG|G|ML|%)', r'\1 \2', extracted_normalized)

        for known in known_strengths:
            known_normalized = known.upper().strip()

            # Exact match
            if extracted_normalized == known_normalized:
                return True, 0.20

            # Numeric value match (25 MG matches 25MG)
            extracted_num = re.search(r'([\d.]+)', extracted_normalized)
            known_num = re.search(r'([\d.]+)', known_normalized)
            if extracted_num and known_num:
                if float(extracted_num.group(1)) == float(known_num.group(1)):
                    # Same number, check unit compatibility
                    if 'MG' in extracted_normalized and 'MG' in known_normalized:
                        return True, 0.18
                    elif 'MCG' in extracted_normalized and 'MCG' in known_normalized:
                        return True, 0.18
                    # Number matches but can't confirm unit
                    return True, 0.10

        return False, 0.0

    def check_medication_has_strength(self, medication_name: str, strength: str) -> bool:
        """
        Quick check if any medication matching name has the given strength.

        Useful for validating OCR candidates.

        Args:
            medication_name: Medication name
            strength: Strength to check

        Returns:
            True if any match has this strength
        """
        if not self._available or not strength:
            return False

        try:
            import re
            cursor = self._conn.cursor()

            # Extract numeric value from strength
            strength_num = re.search(r'([\d.]+)', strength)
            if not strength_num:
                return False

            strength_val = strength_num.group(1)

            cursor.execute(
                """
                SELECT 1 FROM medications m
                JOIN medication_dosages d ON m.rxcui = d.rxcui
                WHERE m.name_lower LIKE ?
                AND d.strength LIKE ?
                LIMIT 1
                """,
                ('%' + medication_name.lower() + '%', '%' + strength_val + '%')
            )
            return cursor.fetchone() is not None
        except Exception as e:
            logger.debug(f"Strength check failed: {e}")
            return False

    def lookup_ocr_tolerant(
        self,
        medication_name: str,
        max_distance: int = None,
        extracted_strength: str = None,
        context_medications: List[str] = None
    ) -> Optional[dict]:
        """
        OCR-error-tolerant medication lookup using edit distance, phonetic matching,
        dosage validation, and prescription context.

        Handles common OCR errors in handwritten prescriptions:
        - Missing/extra letters (Atenolol -> Aerol, Aenolol)
        - Character substitutions (m->rn, t->l, n->u)
        - Transpositions (Metformin -> Metfromin)

        Enhanced with:
        - Dosage validation: Boosts candidates whose known strengths match extracted strength
        - Context scoring: Uses other medications on prescription to infer drug class
        - OCR variant generation: Tries common character confusion patterns

        Args:
            medication_name: Potentially OCR-corrupted medication name
            max_distance: Maximum Levenshtein distance (default: dynamic based on length)
            extracted_strength: Strength from prescription (e.g., "25mg") for validation
            context_medications: Other medications on same prescription for class inference

        Returns:
            Dict with rxcui, name, term_type, match_type, edit_distance, etc. or None
        """
        if not self._available:
            return None

        name_lower = medication_name.lower().strip()
        logger.debug(f"OCR-tolerant lookup for: '{name_lower}'")

        # Skip very short names
        if len(name_lower) < 4:
            logger.debug(f"Skipping '{name_lower}' - too short (< 4 chars)")
            return None

        # Dynamic max distance based on word length
        # Longer words can tolerate more errors
        # INCREASED thresholds for handwritten prescriptions with severe OCR errors
        # e.g., "aerol" -> "atenolol" needs distance 4
        if max_distance is None:
            if len(name_lower) <= 4:
                max_distance = 3
            elif len(name_lower) <= 5:
                max_distance = 4  # Increased from 3
            elif len(name_lower) <= 7:
                max_distance = 5  # Increased from 4
            else:
                max_distance = 6  # Increased from 5

        try:
            cursor = self._conn.cursor()

            # Wider length range to catch OCR errors that drop/add characters
            # For handwritten prescriptions, OCR can miss many characters
            # e.g., "aerol" (5 chars) should match "atenolol" (8 chars)
            min_len = max(3, len(name_lower) - max_distance)
            max_len = len(name_lower) + max_distance + 4  # Extra room for missing chars

            candidates = []

            # Strategy 1: Same first letter, relaxed length
            # Include more term_types: IN (Ingredient), BN (Brand Name), PIN, MIN,
            # SCD (Semantic Clinical Drug), SCDF (Semantic Clinical Drug Form),
            # SBD (Semantic Branded Drug), SBDF (Semantic Branded Drug Form)
            cursor.execute(
                """
                SELECT DISTINCT rxcui, name, name_lower, term_type
                FROM medications
                WHERE name_lower LIKE ?
                AND LENGTH(name_lower) BETWEEN ? AND ?
                AND term_type IN ('IN', 'BN', 'PIN', 'MIN', 'SCD', 'SCDF', 'SBD', 'SBDF', 'GPCK', 'BPCK')
                LIMIT 500
                """,
                (name_lower[0] + '%', min_len, max_len)
            )
            candidates.extend(cursor.fetchall())

            # Strategy 2: Same first letter AND same pharmaceutical suffix
            pharma_suffixes = ['olol', 'ol', 'in', 'ide', 'ate', 'one', 'ine', 'pril', 'sartan',
                               'statin', 'mycin', 'cillin', 'zole', 'pam', 'lam']
            detected_suffix = None
            for suffix in pharma_suffixes:
                if name_lower.endswith(suffix):
                    detected_suffix = suffix
                    # Search for meds with same first letter AND same suffix
                    cursor.execute(
                        """
                        SELECT DISTINCT rxcui, name, name_lower, term_type
                        FROM medications
                        WHERE name_lower LIKE ?
                        AND name_lower LIKE ?
                        AND LENGTH(name_lower) BETWEEN ? AND ?
                        AND term_type IN ('IN', 'BN', 'PIN', 'MIN', 'SCD', 'SCDF', 'SBD', 'SBDF')
                        LIMIT 300
                        """,
                        (name_lower[0] + '%', '%' + suffix, min_len, max_len)
                    )
                    candidates.extend(cursor.fetchall())
                    break

            # Strategy 3: Just same suffix (for when first letter is misread)
            if detected_suffix:
                cursor.execute(
                    """
                    SELECT DISTINCT rxcui, name, name_lower, term_type
                    FROM medications
                    WHERE name_lower LIKE ?
                    AND LENGTH(name_lower) BETWEEN ? AND ?
                    AND term_type IN ('IN', 'BN', 'PIN', 'MIN', 'SCD', 'SCDF', 'SBD', 'SBDF')
                    LIMIT 300
                    """,
                    ('%' + detected_suffix, min_len, max_len)
                )
                candidates.extend(cursor.fetchall())

            # Strategy 4: Try OCR confusion variants
            ocr_variants = generate_ocr_variants(name_lower)
            for variant in ocr_variants[:10]:  # Limit variants checked
                cursor.execute(
                    """
                    SELECT DISTINCT rxcui, name, name_lower, term_type
                    FROM medications
                    WHERE name_lower = ?
                    AND term_type IN ('IN', 'BN', 'PIN', 'MIN', 'SCD', 'SCDF', 'SBD', 'SBDF')
                    LIMIT 10
                    """,
                    (variant,)
                )
                candidates.extend(cursor.fetchall())

            # Strategy 5: If suffix suggests beta-blocker (ol, olol), search common beta-blockers directly
            # This handles severe OCR errors like "aerol" -> "atenolol"
            if detected_suffix in ('ol', 'olol'):
                common_beta_blockers = [
                    'atenolol', 'metoprolol', 'propranolol', 'carvedilol', 'bisoprolol',
                    'labetalol', 'nadolol', 'nebivolol', 'acebutolol', 'betaxolol',
                    'esmolol', 'sotalol', 'timolol', 'pindolol'
                ]
                for med in common_beta_blockers:
                    cursor.execute(
                        """
                        SELECT DISTINCT rxcui, name, name_lower, term_type
                        FROM medications
                        WHERE name_lower = ?
                        LIMIT 1
                        """,
                        (med,)
                    )
                    result = cursor.fetchone()
                    if result:
                        candidates.append(result)

            # Strategy 6: Broader search - any medication with similar length regardless of term_type
            # For really garbled OCR, try a broader search
            cursor.execute(
                """
                SELECT DISTINCT rxcui, name, name_lower, term_type
                FROM medications
                WHERE name_lower LIKE ?
                AND LENGTH(name_lower) BETWEEN ? AND ?
                ORDER BY LENGTH(name_lower) ASC
                LIMIT 200
                """,
                (name_lower[0] + '%', min_len, max_len)
            )
            candidates.extend(cursor.fetchall())

            # Infer expected drug class from context medications
            context_classes = set()
            if context_medications:
                for ctx_med in context_medications:
                    ctx_class = infer_drug_class(ctx_med)
                    if ctx_class:
                        context_classes.add(ctx_class)

            # Deduplicate candidates and get popularity counts
            seen = set()
            unique_candidates = []
            rxcui_counts = {}

            # Get popularity counts for candidates (more entries = more common medication)
            rxcui_list = list(set(row['rxcui'] for row in candidates))
            if rxcui_list:
                placeholders = ','.join('?' * len(rxcui_list))
                cursor.execute(
                    f"SELECT rxcui, COUNT(*) as cnt FROM medications WHERE rxcui IN ({placeholders}) GROUP BY rxcui",
                    rxcui_list
                )
                for row in cursor.fetchall():
                    rxcui_counts[row['rxcui']] = row['cnt']

            for row in candidates:
                key = row['rxcui']
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(row)

            logger.debug(
                f"OCR lookup '{name_lower}': found {len(unique_candidates)} unique candidates "
                f"(max_distance={max_distance}, suffix={detected_suffix})"
            )

            # Log if atenolol is in candidates (for debugging)
            atenolol_found = any(c['name_lower'] == 'atenolol' for c in unique_candidates)
            if 'ol' in name_lower[-3:]:
                logger.debug(f"OCR lookup '{name_lower}': atenolol in candidates = {atenolol_found}")

            # Calculate edit distance for each candidate
            best_match = None
            best_distance = max_distance + 1
            best_similarity = 0
            best_score = -1  # Combined score: lower distance + higher popularity
            input_soundex = soundex(name_lower)

            for row in unique_candidates:
                candidate_name = row['name_lower']
                distance = levenshtein_distance(name_lower, candidate_name)

                # Calculate normalized similarity
                max_len_pair = max(len(name_lower), len(candidate_name))
                similarity = 1 - (distance / max_len_pair)

                # Get popularity (number of entries in database)
                popularity = rxcui_counts.get(row['rxcui'], 1)
                popularity_boost = min(popularity / 10, 1.0)  # Cap at 1.0 for 10+ entries

                # Suffix match bonus: reward candidates that share the same suffix pattern
                suffix_bonus = 0
                if detected_suffix:
                    # Check if candidate ends with same suffix
                    if candidate_name.endswith(detected_suffix):
                        suffix_bonus = 0.15
                        # Extra bonus if the characters before suffix are similar
                        input_stem = name_lower[:-len(detected_suffix)] if len(name_lower) > len(detected_suffix) else name_lower
                        cand_stem = candidate_name[:-len(detected_suffix)] if len(candidate_name) > len(detected_suffix) else candidate_name
                        stem_similarity = 1 - (levenshtein_distance(input_stem, cand_stem) / max(len(input_stem), len(cand_stem), 1))
                        suffix_bonus += stem_similarity * 0.1

                # Phonetic similarity bonus
                phonetic_bonus = 0.1 if soundex(candidate_name) == input_soundex else 0

                # Ingredient (IN) type bonus - prefer generic names
                type_bonus = 0.05 if row['term_type'] == 'IN' else 0

                # Dosage validation bonus: boost if extracted strength matches known strengths
                # This is a STRONG signal - if strength validates, likely correct medication
                dosage_bonus = 0.0
                dosage_penalty = 0.0
                if extracted_strength:
                    strength_valid, strength_boost = self.validate_strength(
                        row['rxcui'], extracted_strength
                    )
                    if strength_valid:
                        # Significant boost for strength match (up to 0.40)
                        dosage_bonus = min(strength_boost * 2.0, 0.40)
                    else:
                        # Check if medication has ANY known strengths
                        known_strengths = self.get_medication_strengths(row['rxcui'])
                        if known_strengths:
                            # Has strengths but none match - penalty
                            dosage_penalty = 0.15
                        # No penalty if no strength data available

                # Context class bonus: boost if candidate is in same drug class as other Rx meds
                context_bonus = 0.0
                if context_classes:
                    candidate_class = infer_drug_class(candidate_name)
                    if candidate_class:
                        # Check if candidate is in a related drug class
                        # Same class = big bonus, related classes = smaller bonus
                        if candidate_class in context_classes:
                            context_bonus = 0.15
                        # Common co-prescribed pairs
                        elif (candidate_class == 'beta_blocker' and 'ace_inhibitor' in context_classes) or \
                             (candidate_class == 'ace_inhibitor' and 'beta_blocker' in context_classes):
                            context_bonus = 0.10
                        elif (candidate_class == 'statin' and 'ace_inhibitor' in context_classes) or \
                             (candidate_class == 'ace_inhibitor' and 'statin' in context_classes):
                            context_bonus = 0.10
                        elif (candidate_class == 'ppi' and 'ssri' in context_classes) or \
                             (candidate_class == 'ssri' and 'ppi' in context_classes):
                            context_bonus = 0.08

                # Combined score
                score = (similarity + (popularity_boost * 0.15) + suffix_bonus +
                         phonetic_bonus + type_bonus + dosage_bonus + context_bonus - dosage_penalty)

                # Accept if within max_distance OR similarity >= 0.5
                if distance <= max_distance or similarity >= 0.5:
                    # Log candidate evaluation for debugging
                    if candidate_name == 'atenolol' or distance <= 4:
                        logger.debug(
                            f"OCR candidate '{candidate_name}': distance={distance}, "
                            f"similarity={similarity:.2f}, score={score:.3f}, max_dist={max_distance}"
                        )
                    # Use combined score for ranking
                    if score > best_score or (score == best_score and distance < best_distance):
                        best_distance = distance
                        best_score = score
                        best_match = row
                        best_similarity = similarity

            if best_match:
                logger.debug(
                    f"OCR lookup '{name_lower}': best match = '{best_match['name_lower']}' "
                    f"(distance={best_distance}, similarity={best_similarity:.2f}, score={best_score:.3f})"
                )

            if best_match and (best_distance <= max_distance or best_similarity >= 0.5):
                result = {
                    'rxcui': best_match['rxcui'],
                    'name': best_match['name'],
                    'term_type': best_match['term_type'],
                    'match_type': 'ocr_corrected',
                    'edit_distance': best_distance,
                    'similarity': round(best_similarity, 2),
                    'original_input': medication_name,
                    'score': round(best_score, 3)
                }

                # Add dosage validation info
                if extracted_strength:
                    strength_valid, _ = self.validate_strength(
                        best_match['rxcui'], extracted_strength
                    )
                    result['strength_validated'] = strength_valid
                    if strength_valid:
                        result['known_strengths'] = self.get_medication_strengths(
                            best_match['rxcui']
                        )[:5]  # Top 5 strengths

                # Add context info
                if context_classes:
                    candidate_class = infer_drug_class(best_match['name_lower'])
                    if candidate_class:
                        result['drug_class'] = candidate_class
                        result['context_match'] = candidate_class in context_classes

                return result

            return None

        except Exception as e:
            logger.error(f"OCR-tolerant lookup failed: {e}")
            return None

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._available = False


# Singleton instance
_db_instance: Optional[MedicationDatabase] = None


def get_medication_db() -> MedicationDatabase:
    """Get the singleton medication database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = MedicationDatabase()
    return _db_instance


def is_valid_medication(medication_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a string is a valid medication name.

    Convenience function that uses the singleton database.

    Args:
        medication_name: Name to validate

    Returns:
        Tuple of (is_valid, rxcui)
    """
    db = get_medication_db()
    return db.is_valid_medication(medication_name)


def get_rxcui(medication_name: str) -> Optional[str]:
    """
    Get RxCUI code for a medication name.

    Convenience function that uses the singleton database.

    Args:
        medication_name: Medication name

    Returns:
        RxCUI code or None
    """
    db = get_medication_db()
    return db.get_rxcui(medication_name)


def lookup_medication(
    medication_name: str,
    ocr_correction: bool = True,
    extracted_strength: str = None,
    context_medications: List[str] = None
) -> Optional[dict]:
    """
    Look up medication info with optional OCR error correction.

    Tries exact/fuzzy match first, then falls back to OCR-tolerant
    matching if no direct match is found.

    Enhanced features when OCR correction is enabled:
    - Dosage validation: Pass extracted_strength to boost candidates with matching strengths
    - Context scoring: Pass other medications on the prescription to infer drug class

    Args:
        medication_name: Medication name to lookup
        ocr_correction: Whether to try OCR error correction (default True)
        extracted_strength: Strength from prescription (e.g., "25mg") for dosage validation
        context_medications: Other medications on same prescription for context scoring

    Returns:
        Dict with rxcui, name, term_type, match_type or None
    """
    db = get_medication_db()

    # Try exact/fuzzy match first
    result = db.lookup_fuzzy(medication_name)
    if result:
        # For exact matches, still add dosage validation info if strength provided
        if extracted_strength:
            strength_valid, _ = db.validate_strength(result['rxcui'], extracted_strength)
            result['strength_validated'] = strength_valid
        return result

    # If no match and OCR correction enabled, try OCR-tolerant lookup
    if ocr_correction:
        result = db.lookup_ocr_tolerant(
            medication_name,
            extracted_strength=extracted_strength,
            context_medications=context_medications
        )
        if result:
            logger.info(
                f"OCR correction: '{medication_name}' -> '{result['name']}' "
                f"(distance: {result['edit_distance']}, similarity: {result['similarity']}, "
                f"score: {result.get('score', 'N/A')})"
            )
            if result.get('strength_validated'):
                logger.info(f"  Strength '{extracted_strength}' validated against known strengths")
            if result.get('context_match'):
                logger.info(f"  Drug class '{result.get('drug_class')}' matches prescription context")
            return result

    return None
