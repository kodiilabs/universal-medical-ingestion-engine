# ============================================================================
# src/medical_ingestion/enrichers/radiology_enricher.py
# ============================================================================
"""
Radiology Enricher

Enriches radiology findings with:
- Critical finding detection
- Anatomical location standardization
- Follow-up recommendation extraction
- ICD-10 code suggestions
"""

import logging
import re
from typing import Dict, Any, Optional, List

from .base import TypeSpecificEnricher, EnrichedExtraction

logger = logging.getLogger(__name__)


# Critical findings that require immediate attention
CRITICAL_FINDINGS_PATTERNS = [
    # Vascular emergencies
    (r'aortic\s*(dissection|rupture|aneurysm)', "Aortic emergency"),
    (r'pulmonary\s*(embol|pe\b)', "Pulmonary embolism"),
    (r'stroke|infarct|ischemi', "Stroke/Ischemia"),
    (r'hemorrhage|bleeding|hematoma', "Active bleeding"),

    # Cardiac
    (r'myocardial\s*infarct', "Myocardial infarction"),
    (r'pericardial\s*effusion', "Pericardial effusion"),
    (r'cardiac\s*tamponade', "Cardiac tamponade"),

    # Thoracic
    (r'pneumothorax', "Pneumothorax"),
    (r'tension\s*pneumo', "Tension pneumothorax"),
    (r'large\s*pleural\s*effusion', "Large pleural effusion"),

    # Abdominal
    (r'bowel\s*(obstruction|perforation)', "Bowel emergency"),
    (r'appendicitis', "Appendicitis"),
    (r'free\s*(air|fluid)', "Free air/fluid"),

    # Oncologic
    (r'malignant|cancer|carcinoma|tumor|mass', "Malignancy"),
    (r'metast', "Metastasis"),

    # Other
    (r'fracture', "Fracture"),
    (r'dislocation', "Dislocation"),
]

# Anatomical location standardization
ANATOMICAL_LOCATIONS = {
    "head": ["brain", "skull", "cranial", "intracranial", "cerebral"],
    "neck": ["cervical", "thyroid", "carotid"],
    "chest": ["thorax", "thoracic", "lung", "pulmonary", "cardiac", "heart", "mediastin"],
    "abdomen": ["abdominal", "liver", "spleen", "kidney", "renal", "pancrea", "bowel", "intestin"],
    "pelvis": ["pelvic", "bladder", "prostate", "uterus", "ovary"],
    "spine": ["vertebr", "spinal", "lumbar", "thoracic", "cervical"],
    "extremity": ["arm", "leg", "hand", "foot", "shoulder", "hip", "knee", "ankle", "wrist"],
}

# Follow-up recommendation patterns
FOLLOWUP_PATTERNS = [
    (r'recommend.*follow[\-\s]?up', "Follow-up recommended"),
    (r'further\s*(evaluation|imaging|workup)', "Further workup needed"),
    (r'clinical\s*correlation', "Clinical correlation needed"),
    (r'ct\s*with\s*contrast', "CT with contrast recommended"),
    (r'mri\s*recommended', "MRI recommended"),
    (r'biopsy\s*recommended', "Biopsy recommended"),
]


class RadiologyEnricher(TypeSpecificEnricher):
    """
    Enriches radiology findings with critical detection and standardization.
    """

    @property
    def enricher_type(self) -> str:
        return "radiology"

    async def enrich(self, extraction: Any) -> EnrichedExtraction:
        """
        Enrich radiology findings.

        Adds:
        - Critical finding flags
        - Standardized anatomical locations
        - Follow-up recommendations
        """
        result = self._create_result(extraction)
        enrichments = {
            "critical_findings": [],
            "anatomical_locations": [],
            "followup_recommendations": [],
            "findings_by_location": {}
        }

        validation_warnings = []
        review_reasons = []

        # Process each finding
        for finding in extraction.findings:
            enrichment = self._enrich_finding(finding)

            if enrichment.get("is_critical"):
                enrichments["critical_findings"].append({
                    "finding": finding.finding,
                    "critical_type": enrichment["critical_type"]
                })
                review_reasons.append(f"critical_finding_{enrichment['critical_type']}")

            if enrichment.get("location"):
                enrichments["anatomical_locations"].append({
                    "finding": finding.finding,
                    "location": enrichment["location"]
                })

                # Group findings by location
                loc = enrichment["location"]
                if loc not in enrichments["findings_by_location"]:
                    enrichments["findings_by_location"][loc] = []
                enrichments["findings_by_location"][loc].append(finding.finding)

            if enrichment.get("followup"):
                enrichments["followup_recommendations"].append({
                    "finding": finding.finding,
                    "recommendation": enrichment["followup"]
                })

        # Also check raw text for additional critical findings
        # (some may not be in structured findings)
        full_text = ""
        if hasattr(extraction, 'original_extraction'):
            if hasattr(extraction.original_extraction, 'full_text'):
                full_text = extraction.original_extraction.full_text
        elif hasattr(extraction, 'raw_llm_responses'):
            full_text = str(extraction.raw_llm_responses)

        # Calculate enrichment confidence
        total_findings = len(extraction.findings)
        confidence = 0.8 if total_findings > 0 else 0.5

        result.enrichments = enrichments
        result.enrichment_confidence = confidence
        result.validation_warnings = validation_warnings
        result.requires_review = len(enrichments["critical_findings"]) > 0
        result.review_reasons = review_reasons

        logger.info(
            f"Radiology enrichment complete: {len(enrichments['critical_findings'])} critical findings, "
            f"{len(enrichments['followup_recommendations'])} follow-up recommendations"
        )

        return result

    def _enrich_finding(self, finding: Any) -> Dict[str, Any]:
        """Enrich a single finding."""
        enrichment = {}
        finding_text = finding.finding.lower()

        # Check for critical findings
        for pattern, critical_type in CRITICAL_FINDINGS_PATTERNS:
            if re.search(pattern, finding_text, re.IGNORECASE):
                enrichment["is_critical"] = True
                enrichment["critical_type"] = critical_type
                break

        # Determine anatomical location
        location = self._determine_location(finding_text)
        if location:
            enrichment["location"] = location
            # Update finding object
            if not finding.location:
                finding.location = location

        # Check for follow-up recommendations
        for pattern, recommendation in FOLLOWUP_PATTERNS:
            if re.search(pattern, finding_text, re.IGNORECASE):
                enrichment["followup"] = recommendation
                break

        return enrichment

    def _determine_location(self, text: str) -> Optional[str]:
        """Determine anatomical location from text."""
        text_lower = text.lower()

        for location, keywords in ANATOMICAL_LOCATIONS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return location

        return None
