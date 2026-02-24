# ============================================================================
# src/medical_ingestion/enrichers/pathology_enricher.py
# ============================================================================
"""
Pathology Enricher

Enriches pathology findings with:
- Cancer staging information
- Margin status detection
- Specimen adequacy assessment
- ICD-O morphology codes
"""

import logging
import re
from typing import Dict, Any, Optional, List

from .base import TypeSpecificEnricher, EnrichedExtraction

logger = logging.getLogger(__name__)


# TNM staging patterns
TNM_PATTERNS = {
    "T": r'\bT([0-4]|is|x)\b',  # Tumor size
    "N": r'\bN([0-3]|x)\b',      # Node involvement
    "M": r'\bM([0-1]|x)\b',      # Metastasis
}

# Grade patterns
GRADE_PATTERNS = [
    (r'grade\s*([1-4IViv]+)', "Tumor grade"),
    (r'gleason\s*(\d+)', "Gleason score"),
    (r'differentiat\w*\s*(well|moderate|poor)', "Differentiation"),
]

# Margin status patterns
MARGIN_PATTERNS = [
    (r'margin[s]?\s*(negative|clear|free)', "negative"),
    (r'margin[s]?\s*(positive|involved)', "positive"),
    (r'(negative|clear|free)\s*margin', "negative"),
    (r'(positive|involved)\s*margin', "positive"),
    (r'close\s*margin', "close"),
]

# Specimen adequacy patterns
ADEQUACY_PATTERNS = [
    (r'adequate\s*(for|specimen)', "adequate"),
    (r'inadequate|insufficient', "inadequate"),
    (r'satisfactory\s*for', "adequate"),
    (r'unsatisfactory', "inadequate"),
]

# Common pathology diagnoses
DIAGNOSIS_PATTERNS = [
    (r'adenocarcinoma', "Adenocarcinoma"),
    (r'squamous\s*cell\s*carcinoma', "Squamous cell carcinoma"),
    (r'melanoma', "Melanoma"),
    (r'lymphoma', "Lymphoma"),
    (r'sarcoma', "Sarcoma"),
    (r'carcinoma\s*in\s*situ', "Carcinoma in situ"),
    (r'dysplasia', "Dysplasia"),
    (r'metastatic', "Metastatic disease"),
    (r'benign', "Benign"),
    (r'malignant', "Malignant"),
]


class PathologyEnricher(TypeSpecificEnricher):
    """
    Enriches pathology findings with staging and diagnostic information.
    """

    @property
    def enricher_type(self) -> str:
        return "pathology"

    async def enrich(self, extraction: Any) -> EnrichedExtraction:
        """
        Enrich pathology findings.

        Adds:
        - TNM staging
        - Tumor grade
        - Margin status
        - Specimen adequacy
        """
        result = self._create_result(extraction)
        enrichments = {
            "staging": {},
            "grade": None,
            "margin_status": None,
            "specimen_adequacy": None,
            "diagnoses": [],
            "requires_additional_stains": False
        }

        validation_warnings = []
        review_reasons = []

        # Combine all findings text for analysis
        all_text = " ".join([f.finding for f in extraction.findings])

        # Extract TNM staging
        staging = self._extract_staging(all_text)
        if staging:
            enrichments["staging"] = staging
            if staging.get("M") == "1":
                review_reasons.append("metastatic_disease")

        # Extract grade
        grade = self._extract_grade(all_text)
        if grade:
            enrichments["grade"] = grade

        # Extract margin status
        margin = self._extract_margin_status(all_text)
        if margin:
            enrichments["margin_status"] = margin
            if margin == "positive":
                review_reasons.append("positive_margins")

        # Extract specimen adequacy
        adequacy = self._extract_adequacy(all_text)
        if adequacy:
            enrichments["specimen_adequacy"] = adequacy
            if adequacy == "inadequate":
                review_reasons.append("inadequate_specimen")
                validation_warnings.append("Specimen may be inadequate for diagnosis")

        # Extract diagnoses
        diagnoses = self._extract_diagnoses(all_text)
        if diagnoses:
            enrichments["diagnoses"] = diagnoses
            if any("malignant" in d.lower() or "carcinoma" in d.lower() for d in diagnoses):
                review_reasons.append("malignancy_detected")

        # Process individual findings
        for finding in extraction.findings:
            enrichment = self._enrich_finding(finding)

            # Update finding with enriched data
            if enrichment.get("severity") and not finding.severity:
                finding.severity = enrichment["severity"]

        # Calculate enrichment confidence
        confidence = 0.8
        if not staging and not grade and not diagnoses:
            confidence = 0.5

        result.enrichments = enrichments
        result.enrichment_confidence = confidence
        result.validation_warnings = validation_warnings
        result.requires_review = len(review_reasons) > 0
        result.review_reasons = review_reasons

        logger.info(
            f"Pathology enrichment complete: staging={bool(staging)}, "
            f"margin={margin}, {len(diagnoses)} diagnoses"
        )

        return result

    def _extract_staging(self, text: str) -> Dict[str, str]:
        """Extract TNM staging from text."""
        staging = {}

        for component, pattern in TNM_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                staging[component] = match.group(1).upper()

        return staging

    def _extract_grade(self, text: str) -> Optional[str]:
        """Extract tumor grade from text."""
        for pattern, grade_type in GRADE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{grade_type}: {match.group(1)}"

        return None

    def _extract_margin_status(self, text: str) -> Optional[str]:
        """Extract margin status from text."""
        for pattern, status in MARGIN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return status

        return None

    def _extract_adequacy(self, text: str) -> Optional[str]:
        """Extract specimen adequacy from text."""
        for pattern, status in ADEQUACY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return status

        return None

    def _extract_diagnoses(self, text: str) -> List[str]:
        """Extract pathology diagnoses from text."""
        diagnoses = []

        for pattern, diagnosis in DIAGNOSIS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if diagnosis not in diagnoses:
                    diagnoses.append(diagnosis)

        return diagnoses

    def _enrich_finding(self, finding: Any) -> Dict[str, Any]:
        """Enrich a single finding."""
        enrichment = {}
        finding_text = finding.finding.lower()

        # Determine severity based on diagnosis
        if any(term in finding_text for term in ["malignant", "carcinoma", "cancer"]):
            enrichment["severity"] = "critical"
        elif any(term in finding_text for term in ["dysplasia", "atypia"]):
            enrichment["severity"] = "abnormal"
        elif "benign" in finding_text:
            enrichment["severity"] = "normal"

        return enrichment
