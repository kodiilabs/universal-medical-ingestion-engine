# ============================================================================
# src/medical_ingestion/core/context/processing_context.py
# ============================================================================
"""
ProcessingContext
- Main context object passed between all agents
- Tracks extracted data, confidence, warnings, audit, FHIR output
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import uuid

from .enums import ConfidenceLevel, ReviewPriority
from .extracted_value import ExtractedValue
import logging
from ..bbox_utils import (
    validate_bbox,
    merge_bboxes,
    find_text_in_word_boxes,
    find_field_value_bbox,
    calculate_bbox_confidence,
    BBoxMatch
)
from .metadata import (
    DocumentMetadata,
    PatientInfo,
    PractitionerInfo,
    OrganizationInfo,
    SpecimenInfo,
    ReportInfo
)


# Field names that should NOT be treated as lab values
# These are metadata fields that sometimes get picked up from table headers
METADATA_FIELD_EXCLUSIONS = {
    # Patient identifiers
    'patient_id', 'patient_name', 'patient', 'name', 'dob', 'date_of_birth',
    'birth_date', 'birthdate', 'age', 'sex', 'gender', 'sex_female', 'sex_male',
    'female', 'male', 'm', 'f', 'patient_number', 'mrn', 'medical_record_number',

    # Account/order identifiers
    'account', 'account_number', 'acct', 'acct_number', 'order_number',
    'order_id', 'accession', 'accession_number', 'requisition', 'requisition_number',

    # Specimen identifiers
    'specimen_id', 'specimen_number', 'specimen', 'sample_id', 'sample_number',
    'collection_date', 'received_date', 'reported_date', 'report_date',

    # Provider/facility
    'physician', 'doctor', 'ordering_physician', 'provider', 'npi',
    'facility', 'lab', 'laboratory', 'clinic', 'hospital', 'address',
    'phone', 'fax', 'client',

    # Header labels that aren't actual tests
    'test', 'test_name', 'result', 'units', 'reference', 'reference_interval',
    'reference_range', 'flag', 'status', 'current', 'previous', 'date',
    'in_range', 'out_of_range', 'normal', 'abnormal', 'critical',

    # Other metadata
    'page', 'page_number', 'total_pages', 'fasting', 'fasting_status',

    # Blood group (not lab values)
    'blood_group', 'blood_type', 'abo_type', 'abo', 'rh_type', 'rh',
    'rh_d_type', 'a', 'b', 'o', 'ab', 'positive', 'negative',
}

# Patterns that indicate a metadata field (partial matches)
METADATA_FIELD_PATTERNS = [
    'patient', 'account', 'specimen', 'accession', 'order',
    'requisition', 'collection_date', 'report_date', 'received_date',
    'physician', 'provider', 'npi', 'facility', 'address',
    'phone', 'fax', 'page_', '_page', 'client_', '_id',
]


@dataclass
class ProcessingContext:
    """
    Complete document processing state.
    """
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_path: Path = None
    timestamp: datetime = field(default_factory=datetime.now)
    document_type: Optional[str] = None
    document_subtype: Optional[str] = None

    template_id: Optional[str] = None
    template_confidence: float = 0.0

    # Legacy patient fields (kept for backward compatibility)
    patient_id: Optional[str] = None
    patient_demographics: Dict[str, Any] = field(default_factory=dict)
    patient_history: List[Dict] = field(default_factory=list)

    # NEW: Structured document metadata
    document_metadata: Optional[DocumentMetadata] = None

    # NEW: Page tracking for multi-page documents
    total_pages: int = 1
    pages_processed: List[int] = field(default_factory=list)
    page_text: Dict[int, str] = field(default_factory=dict)  # Page number -> text content

    extracted_values: List[ExtractedValue] = field(default_factory=list)
    raw_text: str = ""
    sections: Dict[str, Any] = field(default_factory=dict)

    confidence_scores: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    confidence_level: Optional[ConfidenceLevel] = None

    quality_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    critical_findings: List[str] = field(default_factory=list)

    specimen_quality: Dict[str, Any] = field(default_factory=dict)
    specimen_rejected: bool = False
    rejection_reason: Optional[str] = None

    temporal_trends: List[Dict] = field(default_factory=list)
    temporal_flags: List[str] = field(default_factory=list)

    clinical_summary: Optional[str] = None
    reflex_recommendations: List[Dict] = field(default_factory=list)

    fhir_bundle: Optional[Dict] = None
    fhir_validation_errors: List[str] = field(default_factory=list)

    requires_review: bool = False
    review_priority: ReviewPriority = ReviewPriority.LOW
    review_reasons: List[str] = field(default_factory=list)

    agent_executions: List[Dict] = field(default_factory=list)
    processing_steps: List[Dict] = field(default_factory=list)  # Detailed processing steps for workflow visibility
    processing_duration: Optional[float] = None

    # OCR and table data for bounding box lookups
    _ocr_results: Optional[Any] = None  # OCRExtractionResult from ocr_extractor
    _extracted_tables: Optional[List[Any]] = None  # List[ExtractedTable] from table_extractor

    # Image quality analysis (for image-based documents)
    quality_report: Optional[Any] = None  # QualityReport from image_quality

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def add_extracted_value(self, extracted_value: ExtractedValue, deduplicate: bool = True):
        """
        Add an extracted value, with optional deduplication.

        Deduplication logic:
        - If a value with the same field_name already exists, compare quality
        - Keep the better value based on: confidence, completeness (unit, ref range), method
        - This prevents duplicates from multiple extraction methods (template, table, MedGemma)

        Filtering:
        - Metadata fields (patient_id, sex, age, account_number, etc.) are rejected
        - These should go in document_metadata, not extracted_values

        Args:
            extracted_value: The ExtractedValue to add
            deduplicate: If True, check for duplicates and keep better value
        """
        # Filter out metadata fields that shouldn't be lab values
        if self._is_metadata_field(extracted_value.field_name):
            logger = logging.getLogger(__name__)
            logger.debug(f"Skipping metadata field as lab value: {extracted_value.field_name}")
            return

        if deduplicate:
            # Check if we already have this field
            existing_idx = None
            for i, existing in enumerate(self.extracted_values):
                if self._is_same_field(existing.field_name, extracted_value.field_name):
                    existing_idx = i
                    break

            if existing_idx is not None:
                existing = self.extracted_values[existing_idx]

                # Compare quality and keep the better one
                new_score = self._calculate_value_quality(extracted_value)
                existing_score = self._calculate_value_quality(existing)

                if new_score > existing_score:
                    # New value is better, replace
                    self.extracted_values[existing_idx] = extracted_value
                # else: keep existing, don't add new
                return

        # No duplicate found (or deduplication disabled), add the value
        self.extracted_values.append(extracted_value)
        if extracted_value.confidence < self.overall_confidence or self.overall_confidence == 0:
            self.overall_confidence = extracted_value.confidence

    def _is_same_field(self, name1: str, name2: str) -> bool:
        """
        Check if two field names refer to the same test.

        Handles variations like:
        - 'MCV' vs 'mcv' (case)
        - 'red_blood_cell_count' vs 'rbc' (common aliases)
        - 'WBC' vs 'white_blood_cell_count' vs 'wbc_count'
        - 'immature_granulocytes' vs 'immature_granulocytes_abs' (suffix variations)
        """
        # Normalize names
        n1 = name1.lower().replace(' ', '_').replace('-', '_')
        n2 = name2.lower().replace(' ', '_').replace('-', '_')

        # Exact match
        if n1 == n2:
            return True

        # Strip common suffixes that indicate the same test
        # (absolute count, percentage, etc. - these are NOT separate tests)
        suffixes_to_strip = ['_abs', '_absolute', '_count', '_level', '_result']
        n1_stripped = n1
        n2_stripped = n2
        for suffix in suffixes_to_strip:
            if n1.endswith(suffix):
                n1_stripped = n1[:-len(suffix)]
            if n2.endswith(suffix):
                n2_stripped = n2[:-len(suffix)]

        # Check if stripped names match
        if n1_stripped == n2_stripped:
            return True

        # Common lab test aliases
        aliases = {
            # Hematology
            'wbc': ['white_blood_cell_count', 'white_blood_cells', 'wbc_count', 'leukocytes'],
            'rbc': ['red_blood_cell_count', 'red_blood_cells', 'rbc_count', 'erythrocytes'],
            'hgb': ['hemoglobin', 'hb'],
            'hct': ['hematocrit', 'packed_cell_volume', 'pcv'],
            'mcv': ['mean_corpuscular_volume', 'mean_cell_volume'],
            'mch': ['mean_corpuscular_hemoglobin', 'mean_cell_hemoglobin'],
            'mchc': ['mean_corpuscular_hemoglobin_concentration'],
            'rdw': ['red_cell_distribution_width', 'rdw_cv', 'rdw_sd'],
            'plt': ['platelets', 'platelet_count', 'thrombocytes'],
            'mpv': ['mean_platelet_volume'],

            # Differentials
            'neut': ['neutrophils', 'neutrophil_count', 'neut_abs', 'absolute_neutrophils'],
            'lymph': ['lymphocytes', 'lymphocyte_count', 'lymph_abs', 'absolute_lymphocytes'],
            'mono': ['monocytes', 'monocyte_count', 'mono_abs', 'absolute_monocytes'],
            'eos': ['eosinophils', 'eosinophil_count', 'eos_abs', 'absolute_eosinophils'],
            'baso': ['basophils', 'basophil_count', 'baso_abs', 'absolute_basophils'],

            # Chemistry
            'glu': ['glucose', 'blood_glucose', 'fasting_glucose'],
            'bun': ['blood_urea_nitrogen', 'urea_nitrogen'],
            'cr': ['creatinine', 'serum_creatinine'],
            'na': ['sodium', 'serum_sodium'],
            'k': ['potassium', 'serum_potassium'],
            'cl': ['chloride', 'serum_chloride'],
            'co2': ['carbon_dioxide', 'bicarbonate', 'hco3'],
            'ca': ['calcium', 'serum_calcium'],
            'tp': ['total_protein'],
            'alb': ['albumin', 'serum_albumin'],
            'ast': ['aspartate_aminotransferase', 'sgot'],
            'alt': ['alanine_aminotransferase', 'sgpt'],
            'alp': ['alkaline_phosphatase'],
            'tbil': ['total_bilirubin', 'bilirubin_total'],
            'dbil': ['direct_bilirubin', 'bilirubin_direct'],
        }

        # Check if either name matches an alias group
        for canonical, alias_list in aliases.items():
            all_names = [canonical] + alias_list
            n1_matches = any(n1 == a or n1 in a or a in n1 for a in all_names)
            n2_matches = any(n2 == a or n2 in a or a in n2 for a in all_names)
            if n1_matches and n2_matches:
                return True

        return False

    def _calculate_value_quality(self, value: ExtractedValue) -> float:
        """
        Calculate a quality score for an extracted value.

        Higher score = better quality. Factors:
        - Confidence (0-1)
        - Has unit (+0.15)
        - Has reference range (+0.15)
        - Extraction method (template=+0.1, table=+0.05, medgemma=0)
        - Has abnormal flag (+0.05)
        """
        score = value.confidence

        # Bonus for having unit
        if value.unit and value.unit.strip():
            score += 0.15

        # Bonus for having reference range
        if value.reference_min is not None or value.reference_max is not None:
            score += 0.15

        # Bonus based on extraction method
        method = (value.extraction_method or '').lower()
        if 'template' in method:
            score += 0.10
        elif 'table' in method:
            score += 0.05
        # medgemma/llm gets no bonus

        # Bonus for having abnormal flag
        if value.abnormal_flag:
            score += 0.05

        return score

    def _is_metadata_field(self, field_name: str) -> bool:
        """
        Check if a field name is a metadata field that should not be a lab value.

        Metadata fields include patient demographics, account numbers, specimen IDs,
        provider info, etc. These belong in document_metadata, not extracted_values.

        Args:
            field_name: The field name to check

        Returns:
            True if this is a metadata field, False if it's likely a lab value
        """
        if not field_name:
            return False

        # Normalize the field name
        normalized = field_name.lower().strip().replace(' ', '_').replace('-', '_')

        # Check exact matches against known metadata fields
        if normalized in METADATA_FIELD_EXCLUSIONS:
            return True

        # Check for partial pattern matches
        for pattern in METADATA_FIELD_PATTERNS:
            if pattern in normalized:
                return True

        # Check if the "value" looks like an ID (all digits, very long number)
        # This catches cases like "Patient Id" where the field name is the label
        # and the value is a long ID number that got split across columns

        return False

    def sort_by_document_order(self) -> None:
        """
        Sort extracted values to match their order in the original document.

        Uses multiple signals to determine order:
        1. source_page (primary) - page number
        2. source_row_index - explicit row index from table extraction
        3. bbox y-coordinate - vertical position on page
        4. source_location - "Row X" string parsing

        This ensures side-by-side comparison with the original document is intuitive.
        """
        import re

        def get_sort_key(value: ExtractedValue) -> tuple:
            """Generate sort key for document ordering."""
            # Page number (default to 0 if not set)
            page = value.source_page if value.source_page is not None else 0

            # Try multiple methods to get vertical position
            y_position = float('inf')  # Default to end if no position info

            # Method 1: Explicit row index (most reliable from table extraction)
            if value.source_row_index is not None:
                y_position = value.source_row_index

            # Method 2: Bounding box y-coordinate
            elif value.bbox and len(value.bbox) >= 2:
                # bbox is (x0, y0, x1, y1) - use y0 (top of box)
                y_position = value.bbox[1]

            # Method 3: Parse "Row X" from source_location
            elif value.source_location:
                row_match = re.search(r'[Rr]ow\s*(\d+)', value.source_location)
                if row_match:
                    y_position = int(row_match.group(1))

            return (page, y_position)

        self.extracted_values.sort(key=get_sort_key)

    def add_quality_flag(self, flag: str, severity: str = "warning"):
        self.quality_flags.append(flag)
        if severity == "critical":
            self.requires_review = True
            if self.review_priority != ReviewPriority.CRITICAL:
                self.review_priority = ReviewPriority.HIGH

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def add_critical_finding(self, finding: str):
        self.critical_findings.append(finding)
        self.requires_review = True
        self.review_priority = ReviewPriority.CRITICAL

    def log_agent_execution(self, agent_name: str, decision: Dict[str, Any]):
        self.agent_executions.append({
            "agent": agent_name,
            "timestamp": datetime.now(),
            "decision": decision
        })

    def log_processing_step(
        self,
        step_name: str,
        status: str = "completed",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a processing step for workflow visibility.

        Args:
            step_name: Human-readable step name (e.g., "RxNorm Lookup", "OCR Correction")
            status: Step status ("running", "completed", "skipped", "failed")
            details: Optional dict with step-specific details
        """
        self.processing_steps.append({
            "step": step_name,
            "status": status,
            "timestamp": datetime.now(),
            "details": details or {}
        })

    def calculate_confidence_level(self):
        """
        Calculate overall confidence from all components and set confidence level.

        Weights:
        - Classification confidence: 25%
        - Extraction confidence (average): 50%
        - Validation/other scores: 25%
        """
        # Calculate weighted overall confidence
        components = {}
        weights = {
            'classification': 0.25,
            'extraction': 0.50,
            'validation': 0.25
        }

        # Classification confidence
        if 'classification' in self.confidence_scores:
            components['classification'] = self.confidence_scores['classification']

        # Extraction confidence - use average of extracted values, not minimum
        if self.extracted_values:
            extraction_confidences = [v.confidence for v in self.extracted_values if v.confidence > 0]
            if extraction_confidences:
                components['extraction'] = sum(extraction_confidences) / len(extraction_confidences)

        # Validation confidence (if available)
        if 'validation' in self.confidence_scores:
            components['validation'] = self.confidence_scores['validation']
        elif 'extraction' in components:
            # Default validation to extraction if not set
            components['validation'] = components.get('extraction', 0.5)

        # Calculate weighted average
        if components:
            total_weight = 0
            weighted_sum = 0
            for key, value in components.items():
                weight = weights.get(key, 0.1)
                weighted_sum += value * weight
                total_weight += weight

            if total_weight > 0:
                self.overall_confidence = weighted_sum / total_weight

        # Set confidence level based on calculated overall
        if any(v.validation_conflict for v in self.extracted_values):
            self.confidence_level = ConfidenceLevel.CONFLICT
        elif self.overall_confidence >= 0.85:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.overall_confidence >= 0.70:
            self.confidence_level = ConfidenceLevel.MEDIUM
        else:
            self.confidence_level = ConfidenceLevel.LOW

    def should_escalate_to_review(self) -> bool:
        return (
            self.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.CONFLICT]
            or self.specimen_rejected
            or len(self.critical_findings) > 0
            or len(self.quality_flags) > 0
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "template_id": self.template_id,
            "num_extracted_values": len(self.extracted_values),
            "overall_confidence": self.overall_confidence,
            "confidence_level": self.confidence_level,
            "warnings": len(self.warnings),
            "quality_flags": len(self.quality_flags),
            "critical_findings": len(self.critical_findings),
            "requires_review": self.requires_review,
            "review_priority": self.review_priority,
        }

    # ========================================================================
    # BOUNDING BOX HELPERS
    # ========================================================================

    def set_ocr_results(self, ocr_results: Any):
        """Store OCR results for bounding box lookups."""
        self._ocr_results = ocr_results

    def set_extracted_tables(self, tables: List[Any]):
        """Store extracted tables for bounding box lookups."""
        self._extracted_tables = tables

    def find_bbox_for_value(
        self,
        value: str,
        field_name: Optional[str] = None,
        page: Optional[int] = None,
        fuzzy_threshold: float = 0.8
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Find the bounding box for a value by searching through tables and OCR results.

        Uses improved matching with:
        - Fuzzy text matching for OCR errors
        - Field name context for better accuracy
        - Multi-word value support
        - Numeric value normalization

        Args:
            value: The value to search for
            field_name: Optional field/test name to narrow search
            page: Optional page number to narrow search
            fuzzy_threshold: Minimum similarity for fuzzy matching (0-1)

        Returns:
            Normalized bbox (x0, y0, x1, y1) with coordinates in 0-1 range, or None
        """
        if not value:
            return None

        value_str = str(value).strip()
        best_match: Optional[BBoxMatch] = None

        # Search in extracted tables first (more accurate bboxes)
        if self._extracted_tables:
            for table in self._extracted_tables:
                # Filter by page if specified
                table_page = getattr(table, 'page_number', None)
                if page is not None and table_page is not None and table_page != page:
                    continue

                # Try table's find_value_bbox method if available
                if hasattr(table, 'find_value_bbox'):
                    bbox = table.find_value_bbox(value_str)
                    if bbox and validate_bbox(bbox):
                        confidence = calculate_bbox_confidence(bbox, "table")
                        if not best_match or confidence > best_match.confidence:
                            best_match = BBoxMatch(
                                bbox=bbox,
                                text=value_str,
                                confidence=confidence,
                                source="table",
                                match_type="exact",
                                page=table_page
                            )

                # Try cell boxes for more precise matching
                if hasattr(table, 'cell_boxes') and table.cell_boxes:
                    for cell in table.cell_boxes:
                        if hasattr(cell, 'text') and hasattr(cell, 'bbox'):
                            cell_text = str(cell.text).strip()
                            if cell_text.lower() == value_str.lower():
                                if cell.bbox and validate_bbox(cell.bbox):
                                    confidence = calculate_bbox_confidence(cell.bbox, "table")
                                    if not best_match or confidence > best_match.confidence:
                                        best_match = BBoxMatch(
                                            bbox=cell.bbox,
                                            text=cell_text,
                                            confidence=confidence,
                                            source="table",
                                            match_type="exact",
                                            page=table_page
                                        )

        # Search in OCR results with improved matching
        if self._ocr_results and hasattr(self._ocr_results, 'pages'):
            for page_idx, ocr_page in enumerate(self._ocr_results.pages):
                # Get page number (fallback to index + 1)
                ocr_page_num = getattr(ocr_page, 'page_number', page_idx + 1)

                # Filter by page if specified
                if page is not None and ocr_page_num != page:
                    continue

                # Search word boxes with improved matching
                if hasattr(ocr_page, 'word_boxes') and ocr_page.word_boxes:
                    # Use field name context if available for better accuracy
                    if field_name:
                        match = find_field_value_bbox(
                            field_name=field_name,
                            value=value_str,
                            word_boxes=ocr_page.word_boxes,
                            row_tolerance=0.02
                        )
                        if match:
                            # Normalize bbox if needed
                            bbox = match.bbox
                            if ocr_page.page_width and ocr_page.page_height:
                                # Check if normalization is needed (coords > 1)
                                if any(c > 1 for c in bbox):
                                    bbox = (
                                        bbox[0] / ocr_page.page_width,
                                        bbox[1] / ocr_page.page_height,
                                        bbox[2] / ocr_page.page_width,
                                        bbox[3] / ocr_page.page_height
                                    )
                            if validate_bbox(bbox):
                                confidence = match.confidence * calculate_bbox_confidence(bbox, "ocr")
                                if not best_match or confidence > best_match.confidence:
                                    best_match = BBoxMatch(
                                        bbox=bbox,
                                        text=match.text,
                                        confidence=confidence,
                                        source="ocr",
                                        match_type=match.match_type,
                                        page=ocr_page_num
                                    )
                    else:
                        # Fall back to general text search
                        matches = find_text_in_word_boxes(
                            value_str,
                            ocr_page.word_boxes,
                            fuzzy_threshold=fuzzy_threshold
                        )
                        for match in matches:
                            bbox = match.bbox
                            # Normalize if needed
                            if ocr_page.page_width and ocr_page.page_height:
                                if any(c > 1 for c in bbox):
                                    bbox = (
                                        bbox[0] / ocr_page.page_width,
                                        bbox[1] / ocr_page.page_height,
                                        bbox[2] / ocr_page.page_width,
                                        bbox[3] / ocr_page.page_height
                                    )
                            if validate_bbox(bbox):
                                confidence = match.confidence * calculate_bbox_confidence(bbox, "ocr")
                                if not best_match or confidence > best_match.confidence:
                                    best_match = BBoxMatch(
                                        bbox=bbox,
                                        text=match.text,
                                        confidence=confidence,
                                        source="ocr",
                                        match_type=match.match_type,
                                        page=ocr_page_num
                                    )

        return best_match.bbox if best_match else None

    def find_bbox_for_extracted_value(
        self,
        extracted_value: ExtractedValue
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Find the bounding box for an ExtractedValue if not already set.

        This is a convenience method that extracts the value and uses
        find_bbox_for_value with the appropriate parameters.

        Args:
            extracted_value: The ExtractedValue to find bbox for

        Returns:
            Normalized bbox or None
        """
        # If already has bbox, return it
        if extracted_value.bbox:
            return extracted_value.bbox

        # Search for the value
        return self.find_bbox_for_value(
            value=str(extracted_value.value),
            field_name=extracted_value.field_name,
            page=extracted_value.source_page
        )

    def populate_missing_bboxes(self) -> int:
        """
        Populate bounding boxes for all extracted values that don't have them.

        This is useful to call after extraction is complete to fill in
        any missing bounding boxes using OCR and table data.

        Returns:
            Number of bboxes successfully populated
        """
        populated_count = 0
        for extracted_value in self.extracted_values:
            # Validate existing bbox if present
            if extracted_value.bbox:
                if not validate_bbox(extracted_value.bbox):
                    # Invalid bbox, try to find a new one
                    extracted_value.bbox = None

            if not extracted_value.bbox:
                bbox = self.find_bbox_for_extracted_value(extracted_value)
                if bbox and validate_bbox(bbox):
                    extracted_value.bbox = bbox
                    extracted_value.bbox_normalized = True
                    populated_count += 1

        return populated_count

    def get_bbox_match_details(
        self,
        value: str,
        field_name: Optional[str] = None,
        page: Optional[int] = None
    ) -> Optional[BBoxMatch]:
        """
        Get detailed bbox match information including confidence and source.

        Unlike find_bbox_for_value which returns just the bbox, this method
        returns the full BBoxMatch object with confidence, source, and match type.

        Args:
            value: The value to search for
            field_name: Optional field name for context-aware search
            page: Optional page number to narrow search

        Returns:
            BBoxMatch with full details, or None if not found
        """
        if not value:
            return None

        value_str = str(value).strip()
        best_match: Optional[BBoxMatch] = None

        # Search in tables
        if self._extracted_tables:
            for table in self._extracted_tables:
                table_page = getattr(table, 'page_number', None)
                if page is not None and table_page is not None and table_page != page:
                    continue

                if hasattr(table, 'find_value_bbox'):
                    bbox = table.find_value_bbox(value_str)
                    if bbox and validate_bbox(bbox):
                        confidence = calculate_bbox_confidence(bbox, "table")
                        if not best_match or confidence > best_match.confidence:
                            best_match = BBoxMatch(
                                bbox=bbox,
                                text=value_str,
                                confidence=confidence,
                                source="table",
                                match_type="exact",
                                page=table_page
                            )

        # Search in OCR
        if self._ocr_results and hasattr(self._ocr_results, 'pages'):
            for page_idx, ocr_page in enumerate(self._ocr_results.pages):
                ocr_page_num = getattr(ocr_page, 'page_number', page_idx + 1)
                if page is not None and ocr_page_num != page:
                    continue

                if hasattr(ocr_page, 'word_boxes') and ocr_page.word_boxes:
                    if field_name:
                        match = find_field_value_bbox(
                            field_name=field_name,
                            value=value_str,
                            word_boxes=ocr_page.word_boxes
                        )
                    else:
                        matches = find_text_in_word_boxes(value_str, ocr_page.word_boxes)
                        match = matches[0] if matches else None

                    if match:
                        bbox = match.bbox
                        # Normalize if needed
                        if ocr_page.page_width and ocr_page.page_height and any(c > 1 for c in bbox):
                            bbox = (
                                bbox[0] / ocr_page.page_width,
                                bbox[1] / ocr_page.page_height,
                                bbox[2] / ocr_page.page_width,
                                bbox[3] / ocr_page.page_height
                            )
                        if validate_bbox(bbox):
                            confidence = match.confidence * calculate_bbox_confidence(bbox, "ocr")
                            if not best_match or confidence > best_match.confidence:
                                best_match = BBoxMatch(
                                    bbox=bbox,
                                    text=match.text,
                                    confidence=confidence,
                                    source="ocr",
                                    match_type=match.match_type,
                                    page=ocr_page_num
                                )

        return best_match

    # ========================================================================
    # METADATA HELPERS
    # ========================================================================

    def ensure_metadata(self) -> DocumentMetadata:
        """Ensure document_metadata exists and return it."""
        if self.document_metadata is None:
            self.document_metadata = DocumentMetadata()
        return self.document_metadata

    def set_patient_info(self, patient: PatientInfo):
        """Set patient information in document metadata."""
        metadata = self.ensure_metadata()
        metadata.patient = patient
        # Also update legacy fields for backward compatibility
        if patient.patient_id:
            self.patient_id = patient.patient_id
        if patient.age is not None:
            self.patient_demographics['age'] = patient.age
        if patient.sex:
            self.patient_demographics['sex'] = patient.sex

    def set_specimen_info(self, specimen: SpecimenInfo):
        """Set specimen information in document metadata."""
        metadata = self.ensure_metadata()
        metadata.specimen = specimen

    def set_report_info(self, report: ReportInfo):
        """Set report information in document metadata."""
        metadata = self.ensure_metadata()
        metadata.report = report

    def add_practitioner(self, practitioner: PractitionerInfo):
        """Add practitioner information based on role."""
        metadata = self.ensure_metadata()
        if practitioner.role == "ordering":
            metadata.ordering_provider = practitioner
        elif practitioner.role == "lab_director":
            metadata.lab_director = practitioner
        else:
            metadata.other_practitioners.append(practitioner)

    def add_organization(self, organization: OrganizationInfo):
        """Add organization information based on role."""
        metadata = self.ensure_metadata()
        if organization.role == "performing":
            metadata.performing_lab = organization
        elif organization.role == "ordering_facility":
            metadata.ordering_facility = organization
        else:
            metadata.other_organizations.append(organization)

    def get_patient_age_sex(self) -> Tuple[Optional[int], Optional[str]]:
        """Get patient age and sex from metadata or legacy fields."""
        # Try new metadata first
        if self.document_metadata and self.document_metadata.patient:
            patient = self.document_metadata.patient
            return patient.age, patient.sex
        # Fall back to legacy fields
        return (
            self.patient_demographics.get('age'),
            self.patient_demographics.get('sex')
        )

    def get_collection_date(self) -> Optional[Any]:
        """Get specimen collection date from metadata."""
        if self.document_metadata:
            if self.document_metadata.specimen and self.document_metadata.specimen.collection_date:
                return self.document_metadata.specimen.collection_date
            if self.document_metadata.report and self.document_metadata.report.collection_date:
                return self.document_metadata.report.collection_date
        return None

    def add_page_text(self, page_num: int, text: str):
        """Add text content for a specific page."""
        self.page_text[page_num] = text
        if page_num not in self.pages_processed:
            self.pages_processed.append(page_num)
            self.pages_processed.sort()

    def get_full_text_with_page_markers(self) -> str:
        """
        Get full document text with page markers for multi-page awareness.
        Useful for LLM prompts that need to know page context.
        """
        if not self.page_text:
            return self.raw_text

        parts = []
        for page_num in sorted(self.page_text.keys()):
            parts.append(f"\n=== PAGE {page_num} of {self.total_pages} ===\n")
            parts.append(self.page_text[page_num])

        return "\n".join(parts)
