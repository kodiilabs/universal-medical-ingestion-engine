# ============================================================================
# src/medical_ingestion/classifiers/document_classifier.py
# ============================================================================
"""
Document Classification Agent

Architecture (Option B with Layout-Aware Preprocessing):

1. LAYOUT-AWARE PREPROCESSING (EARLY)
   - Extract text with structure information
   - Detect tables, columns, sections, regions
   - Store layout context for downstream use

2. PRE-VALIDATION (quick reject)
   - Minimum text/content threshold
   - Reject non-medical documents early

3. MEDGEMMA CLASSIFICATION (PRIMARY)
   - Receives: text + layout structure
   - Uses vision/semantic understanding
   - Main decision maker

4. FINGERPRINT BOOST (validate/confirm)
   - Weighted pattern matching to validate MedGemma
   - Layout signatures weighted by definitiveness (0.0-1.0)
   - Negative keywords reduce false positives across all types
   - Boost confidence if agrees

5. RESOLVE CLASSIFICATION (ensemble decision)
   - Agreement → boost confidence
   - Disagreement with significant gap (>0.2) → override with higher confidence
   - Disagreement without gap → trust MedGemma, flag for manual review

Classification targets:
- Lab report
- Radiology report
- Pathology report
- Prescription
- Unknown (fallback)

This is the FIRST agent in the pipeline - everything routes through here.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path
import re

from ..core.context.processing_context import ProcessingContext
from ..core.agent_base import Agent
from ..extractors.universal_text_extractor import UniversalTextExtractor
from ..medgemma.client import create_client
from ..constants import DocumentType
from ..config import threshold_settings


class DocumentClassifier(Agent):
    """
    Classifies medical documents into types for routing.

    Architecture (Option B with Layout-Aware Preprocessing):

    1. LAYOUT-AWARE PREPROCESSING (early)
       - Extract document structure: tables, sections, regions
       - Pass structural context to classifier

    2. PRE-VALIDATION (quick reject)
       - Minimum text/content threshold
       - Reject non-medical documents early

    3. MEDGEMMA CLASSIFICATION (primary)
       - Receives: text + layout structure
       - Main decision maker using vision/semantic understanding

    4. FINGERPRINT BOOST (validate/confirm)
       - Weighted pattern matching to validate MedGemma
       - Layout signatures weighted by definitiveness
       - Boost confidence if agrees

    5. RESOLVE CLASSIFICATION (ensemble decision)
       - Agreement → boost, Disagreement → flag for review
       - Override MedGemma only when fingerprint gap > 0.2
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Use UniversalTextExtractor for better OCR routing (TrOCR for images/scanned PDFs)
        self.text_extractor = UniversalTextExtractor(config)
        self.medgemma = create_client(config)

        # Fingerprint patterns for each document type
        self.patterns = self._load_classification_patterns()
    
    def get_name(self) -> str:
        return "DocumentClassifier"
    
    def _load_classification_patterns(self) -> Dict[str, Dict]:
        """
        Load classification patterns for structural fingerprinting.
        
        Each pattern contains keywords and layout signatures
        that strongly indicate a document type.
        """
        return {
            DocumentType.LAB: {
                "keywords": [
                    "lab", "laboratory", "test results", "specimen",
                    "quest diagnostics", "labcorp", "reference range",
                    "cbc", "complete blood count", "metabolic panel",
                    "lipid panel", "hemoglobin", "hematocrit",
                    "blood draw", "fasting", "collected"
                ],
                "header_patterns": [
                    r"laboratory\s+results?",
                    r"lab\s+report",
                    r"test\s+results?",
                    r"quest\s+diagnostics",
                    r"labcorp"
                ],
                # Weights reflect how definitive each signature is (0.0-1.0)
                "layout_signatures": {
                    "has_reference_ranges": 0.9,
                    "has_lab_units": 0.95,
                    "has_test_value_columns": 0.7,
                    "has_abnormal_flags": 0.6
                },
                "negative_keywords": [
                    "refill", "dispense", "pharmacy", "pharmacist",
                    "sig:", "take", "apply", "capsule", "tablet"
                ]
            },

            DocumentType.RADIOLOGY: {
                "keywords": [
                    "radiology", "imaging", "x-ray", "ct scan", "mri",
                    "ultrasound", "mammogram", "radiologist", "impression",
                    "findings", "comparison", "technique", "indication"
                ],
                "header_patterns": [
                    r"radiology\s+report",
                    r"imaging\s+report",
                    r"x-?ray\s+report",
                    r"ct\s+scan",
                    r"mri\s+report"
                ],
                "layout_signatures": {
                    "has_impression_section": 0.8,
                    "has_findings_section": 0.7,
                    "has_comparison": 0.3
                },
                "negative_keywords": [
                    "reference range", "specimen", "collected",
                    "biopsy", "microscopic", "gross description",
                    "refill", "dispense", "pharmacy", "tablet", "capsule"
                ]
            },

            DocumentType.PATHOLOGY: {
                "keywords": [
                    "pathology", "biopsy", "surgical pathology", "cytology",
                    "diagnosis", "specimen", "gross description",
                    "microscopic", "pathologist", "malignant", "benign"
                ],
                "header_patterns": [
                    r"pathology\s+report",
                    r"surgical\s+pathology",
                    r"biopsy\s+report",
                    r"cytology\s+report"
                ],
                "layout_signatures": {
                    "has_diagnosis_section": 0.5,
                    "has_gross_description": 0.95,
                    "has_microscopic_section": 0.95
                },
                "negative_keywords": [
                    "reference range", "g/dl", "k/ul", "mmol/l",
                    "refill", "dispense", "pharmacy",
                    "x-ray", "ct scan", "mri", "ultrasound", "mammogram"
                ]
            },

            DocumentType.PRESCRIPTION: {
                "keywords": [
                    # Definitive prescription signals
                    "prescription", "rx", "pharmacy", "dispense",
                    "refill", "sig:", "directions", "pharmacist",
                    "npi", "dea", "substitute", "generic", "brand",
                    # Supporting drug/dosage signals
                    "medication", "dosage", "quantity",
                    "capsule", "tablet", "topical", "apply",
                    "once daily", "twice daily", "every",
                    "take", "oral"
                ],
                "header_patterns": [
                    r"prescription",
                    r"\bRx\b",
                    r"medication\s+order",
                    r"prescriber",
                    r"pharmacy",
                    r"dispense"
                ],
                "layout_signatures": {
                    "has_drug_name": 0.8,
                    "has_dosage": 0.6,
                    "has_directions": 0.9,
                    "has_refill_info": 0.95,
                    "has_prescriber_info": 0.3,
                    "has_doctor_header": 0.15,
                    "has_medical_registration": 0.15,
                    "has_patient_vitals": 0.1,
                    "has_clinical_sections": 0.1
                },
                "negative_keywords": [
                    "reference range", "specimen", "collected",
                    "blood draw", "fasting", "hemoglobin", "hematocrit",
                    "biopsy", "microscopic", "gross description"
                ]
            }
        }
    
    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Classify document type using MedGemma-primary architecture.

        Pipeline:
        1. Layout-aware preprocessing (extract structure)
        2. Pre-validation (reject non-medical)
        3. MedGemma classification (primary decision)
        4. Fingerprint boost (validate/confirm)

        Returns:
            {
                "type": DocumentType,
                "confidence": float,
                "reasoning": str,
                "method": "medgemma" | "fingerprint_boost" | "pre_validation"
            }
        """

        # ===== STEP 1: TEXT EXTRACTION (Universal - handles PDFs, images, scans) =====
        extraction_result = None
        if context.raw_text:
            text = context.raw_text
            self.logger.debug("Using pre-extracted text for classification")
        else:
            try:
                # UniversalTextExtractor auto-detects document type and routes to:
                # - Digital PDFs → direct text extraction
                # - Scanned PDFs → DocumentPipeline with TrOCR
                # - Images → DocumentPipeline with TrOCR
                extraction_result = await self.text_extractor.extract(context.document_path)
                text = extraction_result.full_text
                context.raw_text = text
                context.total_pages = extraction_result.page_count
                self.logger.info(
                    f"Extracted {len(text)} chars via {extraction_result.extraction_method} "
                    f"({extraction_result.source_type.value}, {extraction_result.page_count} pages)"
                )
            except Exception as e:
                self.logger.error(f"Text extraction failed: {e}")
                text = ""
                context.raw_text = ""

        # ===== STEP 2: LAYOUT-AWARE PREPROCESSING (EARLY) =====
        # Use layout from extraction if available, otherwise analyze text
        if extraction_result and extraction_result.layout:
            layout_context = self._convert_layout_to_context(extraction_result.layout)
        else:
            layout_context = self._extract_layout_context(text)
        self.logger.debug(f"Layout context: {layout_context['structure_summary']}")

        # Store layout in context for downstream processors
        context.sections['_layout'] = layout_context

        # ===== STEP 3: CONTENT ANALYSIS (NO REJECTION) =====
        # Note: We no longer reject documents here. We always attempt to extract
        # and classify. The validation info is stored but does not block processing.
        validation_result = self._validate_medical_content(text)
        if not validation_result['is_medical']:
            self.logger.info(
                f"Document may not be medical: {validation_result['reason']} - continuing with classification"
            )
            # Store validation note but continue processing
            context.sections['_validation'] = {
                'is_medical': False,
                'reason': validation_result['reason'],
                'needs_review': True
            }

        # ===== STEP 4: MEDGEMMA CLASSIFICATION (PRIMARY) =====
        self.logger.info("Running MedGemma as primary classifier")
        medgemma_result = await self._classify_with_medgemma(text, layout_context)

        # ===== STEP 5: FINGERPRINT BOOST (VALIDATE/CONFIRM) =====
        fingerprint_result = self._classify_by_fingerprint(text)

        # ===== STEP 6: RESOLVE CLASSIFICATION =====
        final_result = self._resolve_classification(medgemma_result, fingerprint_result)

        # Store classification result in context
        context.document_type = final_result['type']
        context.confidence_scores['classification'] = final_result['confidence']

        # Store metadata about classification
        context.sections['_classification'] = {
            'type': final_result['type'],
            'confidence': final_result['confidence'],
            'method': final_result.get('method', 'unknown'),
            'is_new_type': final_result.get('is_new_type', False),
            'needs_processor': final_result.get('needs_processor', False),
            'needs_manual_review': final_result.get('needs_manual_review', False),
            'review_reason': final_result.get('review_reason', ''),
            'needs_validation': final_result['confidence'] < 0.7 or final_result.get('is_new_type', False)
        }

        return final_result

    def _resolve_classification(
        self,
        medgemma_result: Dict[str, Any],
        fingerprint_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Intelligent ensemble decision between MedGemma and fingerprint results.

        Logic:
        1. MedGemma failed → use fingerprint as fallback
        2. Both agree → boost confidence
        3. Disagree with significant gap → override with higher-confidence result
        4. Disagree without significant gap → trust MedGemma, flag for review
        """
        # MedGemma failed → use fingerprint
        if medgemma_result['method'] in ('medgemma_failed', 'error'):
            self.logger.warning(
                f"MedGemma failed, using fingerprint: {fingerprint_result['type']} "
                f"(confidence={fingerprint_result['confidence']:.2f})"
            )
            fingerprint_result['reasoning'] += " (MedGemma unavailable)"
            return fingerprint_result

        final_result = medgemma_result.copy()

        # Agreement → boost confidence
        if fingerprint_result['type'] == medgemma_result['type']:
            boost = min(0.15, fingerprint_result['confidence'] * 0.2)
            final_result['confidence'] = min(0.98, medgemma_result['confidence'] + boost)
            final_result['method'] = "medgemma_confirmed"
            final_result['reasoning'] += f" (fingerprint confirmed: +{boost:.2f})"
            self.logger.info(
                f"Classification confirmed: {final_result['type']} "
                f"(boosted to {final_result['confidence']:.2f})"
            )
            return final_result

        # Disagreement → flag for review
        self.logger.warning(
            f"Classification conflict: MedGemma={medgemma_result['type']} "
            f"({medgemma_result['confidence']:.2f}) vs "
            f"fingerprint={fingerprint_result['type']} "
            f"({fingerprint_result['confidence']:.2f})"
        )

        final_result['needs_manual_review'] = True
        final_result['review_reason'] = (
            f"Classification conflict: MedGemma={medgemma_result['type']} "
            f"({medgemma_result['confidence']:.2f}) vs "
            f"Fingerprint={fingerprint_result['type']} "
            f"({fingerprint_result['confidence']:.2f})"
        )

        # Only override MedGemma if fingerprint is significantly more confident
        confidence_gap = fingerprint_result['confidence'] - medgemma_result['confidence']
        if confidence_gap > 0.2:
            final_result['type'] = fingerprint_result['type']
            final_result['confidence'] = fingerprint_result['confidence']
            final_result['method'] = "fingerprint_override"
            final_result['reasoning'] = (
                f"Fingerprint override: {fingerprint_result['type']} "
                f"({fingerprint_result['confidence']:.2f}) vs MedGemma "
                f"{medgemma_result['type']} ({medgemma_result['confidence']:.2f})"
            )
        else:
            # Trust MedGemma but note the conflict
            final_result['reasoning'] += (
                f" (CONFLICT: fingerprint suggests {fingerprint_result['type']} "
                f"at {fingerprint_result['confidence']:.2f})"
            )

        return final_result
    
    def _classify_by_fingerprint(self, text: str) -> Dict[str, Any]:
        """
        Fast classification using pattern matching.
        
        Scores each document type based on:
        - Keyword presence
        - Header pattern matches
        - Layout signatures
        
        Returns classification with highest score.
        """
        text_lower = text.lower()
        scores = {}
        
        for doc_type, patterns in self.patterns.items():
            score = 0.0
            signals = {}

            # Keyword matching (35% weight)
            # Use capped scoring: 5 strong keywords is already definitive
            keywords = patterns['keywords']
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            keyword_score = min(1.0, keyword_matches / 5)
            signals['keywords'] = keyword_score

            # Negative keyword penalty - if negative keywords are found, reduce score
            negative_keywords = patterns.get('negative_keywords', [])
            negative_matches = sum(1 for kw in negative_keywords if kw in text_lower)
            negative_penalty = min(0.4, negative_matches * 0.1)  # Up to -0.4 penalty
            signals['negative_penalty'] = negative_penalty

            # Header pattern matching (25% weight)
            header_patterns = patterns['header_patterns']
            header_text = text[:500].lower()  # Check first 500 chars
            header_matches = sum(
                1 for pattern in header_patterns
                if re.search(pattern, header_text, re.IGNORECASE)
            )
            header_score = min(1.0, header_matches / max(1, len(header_patterns)))
            signals['headers'] = header_score

            # Layout signature matching (40% weight - structural signals are most reliable)
            layout_score = self._check_layout_signatures(
                text,
                patterns['layout_signatures']
            )
            signals['layout'] = layout_score

            # Calculate weighted score
            # Layout (structure) is most reliable for distinguishing lab vs radiology
            weights = {'keywords': 0.35, 'headers': 0.25, 'layout': 0.40}
            total_score = self.calculate_confidence(signals, weights)

            # Apply negative penalty
            total_score = max(0.0, total_score - negative_penalty)

            scores[doc_type] = {
                'score': total_score,
                'signals': signals
            }
        
        # Get best match
        best_type = max(scores.keys(), key=lambda k: scores[k]['score'])
        best_score = scores[best_type]['score']
        
        return {
            "type": best_type.value,
            "confidence": best_score,
            "reasoning": f"Fingerprint match: {best_type.value} ({best_score:.2f})",
            "method": "fingerprint",
            "all_scores": {k.value: v['score'] for k, v in scores.items()}
        }
    
    def _check_layout_signatures(self, text: str, signatures: Dict[str, float]) -> float:
        """
        Check for layout-specific signatures with weighted scoring.

        Each signature has a weight (0.0-1.0) reflecting how definitive it is
        for the document type. The final score is normalized by total possible weight.
        """
        total_weight = sum(signatures.values())
        earned_weight = 0.0

        for sig_name, weight in signatures.items():
            matched = False

            if sig_name == "has_reference_ranges":
                matched = bool(re.search(r'\d+\.?\d*\s*-\s*\d+\.?\d*', text))

            elif sig_name == "has_test_value_columns":
                lines = text.split('\n')
                table_lines = [l for l in lines if len(l.split()) >= 3]
                matched = len(table_lines) > 5

            elif sig_name == "has_abnormal_flags":
                matched = bool(re.search(r'\b[HL]\b|\bCRITICAL\b', text))

            elif sig_name == "has_lab_units":
                # Lab-specific concentration units (not simple mg/mcg used in prescriptions)
                matched = bool(re.search(
                    r'\b(g/dl|k/ul|mmol/l|mg/dl|mcg/dl|u/l|iu/l|miu/ml|ng/ml|pg/ml|fl|10\^3/ul|10\^6/ul)\b',
                    text, re.IGNORECASE
                ))

            elif sig_name == "has_impression_section":
                matched = bool(re.search(r'impression:|conclusion:', text, re.IGNORECASE))

            elif sig_name == "has_findings_section":
                matched = bool(re.search(r'findings?:', text, re.IGNORECASE))

            elif sig_name == "has_comparison":
                matched = bool(re.search(r'compared? to|stable|unchanged|new', text, re.IGNORECASE))

            elif sig_name == "has_diagnosis_section":
                matched = bool(re.search(r'diagnosis:', text, re.IGNORECASE))

            elif sig_name == "has_gross_description":
                matched = bool(re.search(r'gross\s*(description)?:', text, re.IGNORECASE))

            elif sig_name == "has_microscopic_section":
                matched = bool(re.search(r'microscopic:', text, re.IGNORECASE))

            elif sig_name == "has_drug_name":
                matched = bool(re.search(r'[A-Z][a-z]+\s+\d+\s*mg', text))

            elif sig_name == "has_dosage":
                matched = bool(re.search(r'\d+\s*(mg|mcg|ml|g|units?)', text, re.IGNORECASE))

            elif sig_name == "has_directions":
                matched = bool(re.search(
                    r'sig:|take\s+\d+|once\s+daily|twice\s+daily|every\s+\d+\s+hours?',
                    text, re.IGNORECASE
                ))

            elif sig_name == "has_refill_info":
                matched = bool(re.search(r'refill|no\s+refill|\d+\s+refills?|prn', text, re.IGNORECASE))

            elif sig_name == "has_prescriber_info":
                matched = bool(re.search(
                    r'npi|dea|\bm\.?d\.?\b|\bd\.?o\.?\b|prescriber|physician|doctor',
                    text, re.IGNORECASE
                ))

            elif sig_name == "has_doctor_header":
                matched = bool(re.search(r'dr\.?\s+[a-z]+', text[:300], re.IGNORECASE))

            elif sig_name == "has_medical_registration":
                matched = bool(re.search(r'reg\.?\s*no\.?|mmc|mmb?c|mbbs|m\.?b\.?b\.?s\.?', text, re.IGNORECASE))

            elif sig_name == "has_patient_vitals":
                matched = bool(re.search(
                    r'weight\s*\(?kg|height\s*\(?cm|b\.?m\.?i\.?\s*[=:]|bp\s*:|blood\s+pressure|mmhg',
                    text, re.IGNORECASE
                ))

            elif sig_name == "has_clinical_sections":
                matched = bool(re.search(
                    r'chief\s+complaint|clinical\s+finding|diagnosis|opd|patient\s+complaint',
                    text, re.IGNORECASE
                ))

            if matched:
                earned_weight += weight

        return earned_weight / total_weight if total_weight > 0 else 0.0

    def _convert_layout_to_context(self, layout) -> Dict[str, Any]:
        """
        Convert UniversalTextExtractor's LayoutInfo to classifier's layout context format.

        Args:
            layout: LayoutInfo from UniversalTextExtractor

        Returns:
            Dict compatible with classifier's layout context
        """
        # Build structure summary
        summary_parts = []
        if layout.has_tables:
            summary_parts.append(f"tabular data ({layout.table_count} tables)")
        if layout.has_sections:
            summary_parts.append(f"sections: {', '.join(layout.section_headers[:5])}")
        if layout.has_handwriting:
            summary_parts.append("handwriting detected")
        if layout.has_reference_ranges:
            summary_parts.append("reference ranges")
        if layout.has_lab_units:
            summary_parts.append("lab units")

        return {
            "has_table_structure": layout.has_tables,
            "has_sections": layout.has_sections,
            "has_header_region": layout.has_headers,
            "column_count": layout.detected_columns,
            "section_headers": layout.section_headers,
            "table_indicators": [f"{layout.table_count} tables"] if layout.has_tables else [],
            "structure_summary": "; ".join(summary_parts) if summary_parts else "unstructured text",
            # Medical-specific layout info
            "has_reference_ranges": layout.has_reference_ranges,
            "has_test_value_columns": layout.has_test_value_columns,
            "has_abnormal_flags": layout.has_abnormal_flags,
            "has_lab_units": layout.has_lab_units,
            "has_handwriting": layout.has_handwriting
        }

    def _extract_layout_context(self, text: str) -> Dict[str, Any]:
        """
        Layout-aware preprocessing to extract document structure.

        Analyzes the document text to identify:
        - Table structures (rows, columns, headers)
        - Section headers and organization
        - Document regions (header, body, footer)
        - Column layout

        This context is passed to MedGemma for better classification.
        """
        lines = text.split('\n')
        layout = {
            "has_table_structure": False,
            "has_sections": False,
            "has_header_region": False,
            "column_count": 1,
            "section_headers": [],
            "table_indicators": [],
            "structure_summary": ""
        }

        # Detect table structure
        # Tables have consistent column patterns with values/units
        table_line_count = 0
        value_unit_pattern = re.compile(
            r'(\d+\.?\d*)\s+(g/dl|k/ul|mmol/l|mg/dl|mg|mcg|ml|%|fl)',
            re.IGNORECASE
        )
        column_separator_pattern = re.compile(r'\t|  {2,}|\|')

        for line in lines:
            # Check for columnar structure (tabs, multiple spaces, or pipes)
            if column_separator_pattern.search(line):
                parts = column_separator_pattern.split(line)
                if len(parts) >= 3:
                    table_line_count += 1
                    layout["column_count"] = max(layout["column_count"], len(parts))

            # Check for value-unit patterns common in lab reports
            if value_unit_pattern.search(line):
                layout["table_indicators"].append("value_unit_row")

        if table_line_count >= 3:
            layout["has_table_structure"] = True
            layout["table_indicators"].append(f"{table_line_count} columnar rows")

        # Detect section headers (all caps or followed by colon)
        section_patterns = [
            r'^([A-Z][A-Z\s]{2,30}):?\s*$',  # ALL CAPS headers
            r'^(#+\s*\w.*)$',  # Markdown-style headers
            r'^(\w[\w\s]+):\s*$',  # Word followed by colon
        ]

        for line in lines[:50]:  # Check first 50 lines for headers
            line = line.strip()
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    header = match.group(1).strip()
                    if len(header) > 2 and header not in layout["section_headers"]:
                        layout["section_headers"].append(header)
                        break

        if layout["section_headers"]:
            layout["has_sections"] = True

        # Detect header region (first few lines with metadata)
        header_indicators = [
            r'\b(date|patient|name|dob|mrn|account)\b',
            r'\b(hospital|clinic|laboratory|medical)\b',
            r'\b(dr\.?|physician|provider)\b',
        ]
        header_text = '\n'.join(lines[:10]).lower()
        header_matches = sum(
            1 for p in header_indicators
            if re.search(p, header_text, re.IGNORECASE)
        )
        if header_matches >= 2:
            layout["has_header_region"] = True

        # Build structure summary for MedGemma
        summary_parts = []
        if layout["has_header_region"]:
            summary_parts.append("document header with metadata")
        if layout["has_table_structure"]:
            summary_parts.append(f"tabular data ({layout['column_count']} columns)")
        if layout["has_sections"]:
            sections_preview = layout["section_headers"][:5]
            summary_parts.append(f"sections: {', '.join(sections_preview)}")

        layout["structure_summary"] = "; ".join(summary_parts) if summary_parts else "unstructured text"

        return layout

    def _validate_medical_content(self, text: str) -> Dict[str, Any]:
        """
        Pre-validate that the document contains medical content.

        This catches non-medical documents early (office photos, logos, etc.)
        before running the full classification pipeline.

        Returns:
            {
                "is_medical": bool,
                "reason": str,
                "medical_signals": int
            }
        """
        text_lower = text.lower()

        # MINIMUM TEXT THRESHOLD
        # Real medical documents have substantial text content
        # Office photos, logos, etc. have minimal text
        MIN_TEXT_LENGTH = 50  # At least 50 chars of text
        MIN_WORDS = 10  # At least 10 words

        word_count = len(text.split())
        if len(text) < MIN_TEXT_LENGTH or word_count < MIN_WORDS:
            return {
                "is_medical": False,
                "reason": f"Insufficient text content ({len(text)} chars, {word_count} words)",
                "medical_signals": 0
            }

        # CONTENT SIGNALS
        # Terms that indicate this is a healthcare-related document
        # Includes medical, insurance, paramedical, and billing documents
        medical_indicators = [
            # Patient/clinical terms
            r'\b(patient|diagnosis|treatment|medication|prescription)\b',
            r'\b(clinical|medical|health|hospital|clinic)\b',
            # Lab-specific
            r'\b(test|result|specimen|laboratory|blood|urine)\b',
            r'\b(reference|range|normal|abnormal|high|low)\b',
            # Measurement units
            r'\b(mg|mcg|ml|g/dl|mmol|units?)\b',
            # Document structure
            r'\b(report|findings|impression|conclusion)\b',
            # Drug-related
            r'\b(dosage|dose|tablet|capsule|sig:|rx)\b',
            # Anatomy/body parts
            r'\b(heart|lung|liver|kidney|brain|bone|tissue)\b',
            # Insurance/paramedical terms
            r'\b(claim|coverage|benefit|insurance|insur[ée]|assuré)\b',
            r'\b(paramedical|psychotherapist|counsell?ing|therapist|physiotherapy)\b',
            r'\b(participant|provider|fournisseur|prestations)\b',
            r'\b(submission|réclamation|santé|mental)\b',
            # Billing/receipt terms
            r'\b(receipt|invoice|payment|amount|subtotal|total)\b',
            r'\b(service|session|appointment|visit)\b',
        ]

        signal_count = 0
        for pattern in medical_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                signal_count += 1

        # NON-MEDICAL INDICATORS (strong negative signals)
        # Be careful not to match things that could appear in medical docs
        non_medical_patterns = [
            r'\b(office\s+photo|workspace|meeting\s+room|conference\s+room)\b',
            r'\b(welcome\s+to|about\s+us|our\s+products)\b',
            r'\b(all\s+rights\s+reserved)\b',
        ]

        non_medical_count = 0
        for pattern in non_medical_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                non_medical_count += 1

        # Decision logic:
        # - Need at least 2 medical signals
        # - OR if non-medical signals outweigh medical signals, reject
        MIN_MEDICAL_SIGNALS = 2

        if signal_count < MIN_MEDICAL_SIGNALS:
            return {
                "is_medical": False,
                "reason": f"No medical content detected ({signal_count} signals)",
                "medical_signals": signal_count
            }

        if non_medical_count > signal_count:
            return {
                "is_medical": False,
                "reason": f"Likely non-medical document ({non_medical_count} non-medical vs {signal_count} medical signals)",
                "medical_signals": signal_count
            }

        return {
            "is_medical": True,
            "reason": "Medical content validated",
            "medical_signals": signal_count
        }

    async def _classify_with_medgemma(
        self,
        text: str,
        layout_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        PRIMARY classification using MedGemma with layout awareness.

        Args:
            text: Extracted document text
            layout_context: Structural information from preprocessing

        Uses medical reasoning + layout structure to determine document type.
        """
        # Truncate text if too long (keep first 2000 chars)
        text_sample = text[:2000]

        # Build layout description for the prompt
        layout_desc = ""
        if layout_context:
            layout_desc = f"""
            Document structure detected:
            - {layout_context.get('structure_summary', 'unknown structure')}
            - Has table structure: {layout_context.get('has_table_structure', False)}
            - Has sections: {layout_context.get('has_sections', False)}
            - Section headers found: {', '.join(layout_context.get('section_headers', [])[:5]) or 'none'}
            """

        prompt = f"""You are a medical document classifier. Analyze this document and determine its type.
{layout_desc}
            Document text:
            {text_sample}

            Return ONLY a JSON object with this exact structure:
            {{
                "type": "<document_type>",
                "confidence": <float between 0 and 1>,
                "reasoning": "<brief explanation>"
            }}

            CLASSIFICATION RULES (use these to distinguish similar document types):
            - lab: MUST have test values with units and reference ranges (e.g., CBC, metabolic panel)
            - radiology: MUST have findings and impression sections for imaging studies
            - pathology: MUST have tissue/specimen analysis with microscopic or gross description
            - prescription: MUST have medication names with dosing instructions (drug + dose + frequency)
            - insurance: Insurance claims, coverage documents, EOBs, paramedical forms
            - clinical_notes: Progress notes, SOAP notes, visit summaries
            - discharge_summary: Hospital discharge documents
            - referral: Referral letters between providers
            - consent: Consent forms, authorization documents
            - billing: Medical bills, invoices, statements
            - receipt: Payment receipts, transaction records

            If the document doesn't fit any known type, create a descriptive type name
            (e.g., "medical_record", "intake_form", "questionnaire", etc.)

            Focus on document structure, data organization, and the type of medical data
            (quantitative vs narrative vs prescriptive).

            JSON response:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            # Extract JSON from response
            result = self.medgemma.extract_json(response['text'])
            
            if result and 'type' in result:
                # Normalize type to lowercase for consistent routing
                doc_type = result['type'].lower().strip().replace(' ', '_')

                # Known types that have specialized processors
                known_types = ['lab', 'radiology', 'pathology', 'prescription']

                # Flag if this is a new/unknown type that may need a processor added
                is_new_type = doc_type not in known_types and doc_type != 'unknown'

                return {
                    "type": doc_type,
                    "confidence": result.get('confidence', 0.5),
                    "reasoning": result.get('reasoning', 'MedGemma classification'),
                    "method": "medgemma",
                    "is_new_type": is_new_type,
                    "needs_processor": is_new_type
                }
            else:
                # JSON extraction failed - return unknown with low confidence
                self.logger.warning("MedGemma classification failed - JSON parse error")
                return {
                    "type": DocumentType.UNKNOWN.value,
                    "confidence": 0.3,
                    "reasoning": "Classification uncertain - failed to parse MedGemma response",
                    "method": "medgemma_failed"
                }
        
        except Exception as e:
            self.logger.error(f"MedGemma classification error: {e}")
            return {
                "type": DocumentType.UNKNOWN.value,
                "confidence": 0.2,
                "reasoning": f"MedGemma error: {str(e)}",
                "method": "error"
            }