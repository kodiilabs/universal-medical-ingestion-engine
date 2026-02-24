# ============================================================================
# src/medical_ingestion/enrichers/prescription_enricher.py
# ============================================================================
"""
Prescription Enricher

Enriches medication data with:
- RxNorm codes (via medication_db.py — full RxNorm database)
- Drug interaction checking
- Dosage validation
- OCR correction for drug names
- validation_status for regulatory audit trail
"""

import logging
import re
from typing import Dict, Any, Optional, List

from .base import TypeSpecificEnricher, EnrichedExtraction

logger = logging.getLogger(__name__)


# Known drug interactions (simplified)
DRUG_INTERACTIONS = {
    ("warfarin", "aspirin"): {
        "severity": "major",
        "description": "Increased bleeding risk"
    },
    ("warfarin", "ibuprofen"): {
        "severity": "major",
        "description": "Increased bleeding risk"
    },
    ("lisinopril", "potassium"): {
        "severity": "moderate",
        "description": "Risk of hyperkalemia"
    },
    ("metformin", "contrast"): {
        "severity": "major",
        "description": "Risk of lactic acidosis with IV contrast"
    },
    ("ssri", "ssri"): {
        "severity": "major",
        "description": "Risk of serotonin syndrome"
    },
}


class PrescriptionEnricher(TypeSpecificEnricher):
    """
    Enriches medication data with RxNorm codes and interaction checks.

    Uses the full RxNorm database via medication_db.py for:
    - Exact name matching
    - Fuzzy/OCR-tolerant matching
    - Drug class inference

    Sets validation_status on each medication:
    - verified: Found in RxNorm (exact/fuzzy/prefix match)
    - ocr_corrected: Found via OCR edit distance correction
    - unverified: Not found in any database — needs human review
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._med_db = None
        self._db_available = None

    def _get_med_db(self):
        """Lazy-load medication database."""
        if self._med_db is None:
            try:
                from ..constants.medication_db import get_medication_db
                self._med_db = get_medication_db()
                self._db_available = self._med_db.is_available
            except Exception as e:
                logger.warning(f"Could not load medication database: {e}")
                self._med_db = None
                self._db_available = False
        return self._med_db

    @property
    def enricher_type(self) -> str:
        return "prescription"

    async def enrich(self, extraction: Any) -> EnrichedExtraction:
        """
        Enrich medication data.

        Adds:
        - RxNorm codes (from full database)
        - Drug class information
        - Interaction warnings
        - OCR corrections
        - validation_status for each medication
        """
        result = self._create_result(extraction)
        enrichments = {
            "rxnorm_codes": [],
            "drug_classes": [],
            "ocr_corrections": [],
            "interactions": [],
            "validation_summary": {}
        }

        validation_warnings = []
        review_reasons = []

        # Collect all medication names for context
        all_med_names = [med.name for med in extraction.medications if med.name]

        # Process each medication
        drug_names = []
        status_counts = {}
        for medication in extraction.medications:
            # Get other meds as context
            context_meds = [n for n in all_med_names if n != medication.name]
            enrichment = self._enrich_medication(medication, context_meds)

            # Track validation status
            vs = medication.validation_status or 'unverified'
            status_counts[vs] = status_counts.get(vs, 0) + 1

            if enrichment.get("rxcui"):
                enrichments["rxnorm_codes"].append({
                    "drug": medication.name,
                    "rxcui": enrichment["rxcui"],
                    "drug_class": enrichment.get("drug_class")
                })
                drug_names.append(medication.name.lower())

            if enrichment.get("corrected_name"):
                enrichments["ocr_corrections"].append({
                    "original": enrichment.get("original_name", medication.name),
                    "corrected": enrichment["corrected_name"]
                })

            if enrichment.get("drug_class"):
                enrichments["drug_classes"].append(enrichment["drug_class"])

            if vs == 'unverified':
                validation_warnings.append(
                    f"Medication '{medication.name}' could not be verified against RxNorm database"
                )
                review_reasons.append(f"unverified_medication_{medication.name}")
            elif vs == 'strength_mismatch':
                warning = enrichment.get("strength_warning", f"Strength mismatch for '{medication.name}'")
                validation_warnings.append(warning)
                review_reasons.append(f"strength_mismatch_{medication.name}")

        enrichments["validation_summary"] = status_counts

        # Check for drug interactions
        interactions = self._check_interactions(drug_names)
        if interactions:
            enrichments["interactions"] = interactions
            for interaction in interactions:
                if interaction["severity"] == "major":
                    review_reasons.append(f"drug_interaction_{interaction['drugs'][0]}_{interaction['drugs'][1]}")

        # Calculate enrichment confidence
        total_meds = len(extraction.medications)
        verified_count = sum(1 for m in extraction.medications if m.validation_status in ('verified', 'ocr_corrected'))
        confidence = verified_count / total_meds if total_meds > 0 else 0.5

        result.enrichments = enrichments
        result.enrichment_confidence = confidence
        result.validation_warnings = validation_warnings
        result.requires_review = len(review_reasons) > 0
        result.review_reasons = review_reasons

        logger.info(
            f"Prescription enrichment complete: "
            f"{' | '.join(f'{c} {s}' for s, c in sorted(status_counts.items()))}, "
            f"{len(interactions)} interactions detected"
        )

        return result

    def _enrich_medication(self, medication: Any, context_meds: List[str] = None) -> Dict[str, Any]:
        """Enrich a single medication using the full RxNorm database."""
        enrichment = {}
        original_name = medication.name
        drug_name = self._clean_drug_name(medication.name)

        # Try the real medication database first
        med_db = self._get_med_db()
        if med_db and self._db_available:
            from ..constants.medication_db import lookup_medication

            lookup_name = drug_name

            db_result = lookup_medication(
                lookup_name,
                extracted_strength=medication.strength,
                context_medications=context_meds,
                ocr_correction=True
            )

            if db_result:
                rxcui = db_result.get('rxcui')
                match_type = db_result.get('match_type', 'exact')

                medication.rxcui = rxcui
                medication.rxnorm_name = db_result.get('name')

                enrichment["rxcui"] = rxcui
                enrichment["rxnorm_name"] = db_result.get('name')
                enrichment["drug_class"] = db_result.get('drug_class')

                # Determine validation_status based on match type
                if match_type == 'ocr_corrected':
                    medication.validation_status = 'ocr_corrected'
                    enrichment["corrected_name"] = db_result.get('name')
                    enrichment["original_name"] = original_name
                    enrichment["edit_distance"] = db_result.get('edit_distance')
                else:
                    medication.validation_status = 'verified'

                # Cross-validate strength against known strengths for this drug
                if medication.strength and rxcui:
                    strength_valid = db_result.get('strength_validated')
                    if strength_valid is None:
                        # Wasn't checked in lookup — check now
                        strength_valid, _ = med_db.validate_strength(rxcui, medication.strength)
                    if not strength_valid:
                        known = med_db.get_medication_strengths(rxcui)
                        if known:
                            # Drug has known strengths but extracted strength doesn't match
                            medication.validation_status = 'strength_mismatch'
                            enrichment["strength_warning"] = (
                                f"Extracted strength '{medication.strength}' not found in known strengths "
                                f"for {db_result.get('name')}: {', '.join(known[:8])}"
                            )
                            logger.warning(
                                f"STRENGTH MISMATCH: '{original_name}' matched to "
                                f"'{db_result.get('name')}' (RXCUI:{rxcui}) but "
                                f"strength '{medication.strength}' not in known strengths: {known[:8]}"
                            )

                logger.debug(
                    f"RxNorm match: '{original_name}' -> "
                    f"RXCUI:{rxcui} ({match_type}, status: {medication.validation_status})"
                )
                return enrichment

            # If cleaned name didn't work, try the raw name
            if lookup_name != original_name:
                db_result = lookup_medication(original_name, extracted_strength=medication.strength, ocr_correction=True)
                if db_result:
                    rxcui = db_result.get('rxcui')
                    medication.rxcui = rxcui
                    medication.rxnorm_name = db_result.get('name')
                    medication.validation_status = 'verified' if db_result.get('match_type') != 'ocr_corrected' else 'ocr_corrected'
                    enrichment["rxcui"] = rxcui
                    enrichment["drug_class"] = db_result.get('drug_class')
                    if db_result.get('match_type') == 'ocr_corrected':
                        enrichment["corrected_name"] = db_result.get('name')
                        enrichment["original_name"] = original_name
                    # Strength cross-validation
                    if medication.strength and rxcui:
                        strength_valid = db_result.get('strength_validated')
                        if strength_valid is None:
                            strength_valid, _ = med_db.validate_strength(rxcui, medication.strength)
                        if not strength_valid:
                            known = med_db.get_medication_strengths(rxcui)
                            if known:
                                medication.validation_status = 'strength_mismatch'
                                enrichment["strength_warning"] = (
                                    f"Extracted strength '{medication.strength}' not found in known strengths "
                                    f"for {db_result.get('name')}: {', '.join(known[:8])}"
                                )
                    return enrichment

        # Database unavailable or no match — mark as unverified
        medication.validation_status = 'unverified'
        logger.warning(
            f"UNVERIFIED medication: '{original_name}' — not found in RxNorm database"
        )

        # Validate dosage format
        if medication.strength:
            if not self._validate_dosage(medication.strength):
                enrichment["dosage_warning"] = "Unusual dosage format"

        return enrichment

    def _clean_drug_name(self, text: str) -> str:
        """
        Clean drug name: strip dosage form prefixes, strength, schedule words.

        Handles international formats (Indian Rx: "TAB. METFORMIN 500mg Morning")
        """
        cleaned = text.strip()

        # Strip leading dosage form prefixes
        cleaned = re.sub(
            r'^(TAB|CAP|CAPS|INJ|SYR|SUSP|OINT|GEL|DRP|DROPS|CR|LOT|LOTION|INHALER|PATCH|SUPP)\.?\s+',
            '', cleaned, flags=re.IGNORECASE
        )

        # Remove trailing schedule/timing words
        cleaned = re.sub(
            r'\s+\d*\s*(morning|evening|night|bedtime|afternoon|before\s+food|after\s+food|empty\s+stomach|sos|stat|prn)$',
            '', cleaned, flags=re.IGNORECASE
        )

        # Remove strength patterns
        cleaned = re.sub(r'\s+\d+(\.\d+)?\s*(mg|mcg|g|ml|units?|%|/\d+).*$', '', cleaned, flags=re.IGNORECASE)

        # Remove trailing bare numbers
        cleaned = re.sub(r'\s+\d+(\.\d+)?$', '', cleaned)

        # Remove common dosage form words at end
        cleaned = re.sub(
            r'\s+(tablet|capsule|cap|tab|pill|cream|ointment|gel|solution|suspension|syrup|injection|patch|inhaler|spray|drops|suppository|liquid|powder)s?$',
            '', cleaned, flags=re.IGNORECASE
        )

        return cleaned.strip() or text

    def _validate_dosage(self, dosage: str) -> bool:
        """Validate dosage format."""
        valid_patterns = [
            r'\d+\s*(mg|mcg|g|ml|units?|iu)',
            r'\d+/\d+',
            r'\d+\.\d+\s*(mg|mcg|g|ml)',
        ]

        dosage_lower = dosage.lower()
        for pattern in valid_patterns:
            if re.search(pattern, dosage_lower):
                return True

        return False

    def _check_interactions(
        self,
        drug_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Check for drug interactions."""
        interactions = []

        # Get drug classes for class-based interactions
        drug_classes = {}
        med_db = self._get_med_db()
        if med_db and self._db_available:
            from ..constants.medication_db import lookup_medication
            for name in drug_names:
                result = lookup_medication(name, ocr_correction=False)
                if result and result.get('drug_class'):
                    drug_classes[name] = result['drug_class']

        # Check pairwise interactions
        for i, drug1 in enumerate(drug_names):
            for drug2 in drug_names[i + 1:]:
                # Check direct interaction
                interaction = self._get_interaction(drug1, drug2)
                if interaction:
                    interactions.append({
                        "drugs": [drug1, drug2],
                        "severity": interaction["severity"],
                        "description": interaction["description"]
                    })
                    continue

                # Check class-based interaction (e.g., two SSRIs)
                class1 = drug_classes.get(drug1)
                class2 = drug_classes.get(drug2)

                if class1 and class2 and class1 == class2:
                    if class1 in ["SSRI", "Benzodiazepine", "Opioid"]:
                        interactions.append({
                            "drugs": [drug1, drug2],
                            "severity": "moderate",
                            "description": f"Multiple {class1} medications - monitor closely"
                        })

        return interactions

    def _get_interaction(
        self,
        drug1: str,
        drug2: str
    ) -> Optional[Dict[str, str]]:
        """Get interaction information for two drugs."""
        if (drug1, drug2) in DRUG_INTERACTIONS:
            return DRUG_INTERACTIONS[(drug1, drug2)]
        if (drug2, drug1) in DRUG_INTERACTIONS:
            return DRUG_INTERACTIONS[(drug2, drug1)]

        return None
