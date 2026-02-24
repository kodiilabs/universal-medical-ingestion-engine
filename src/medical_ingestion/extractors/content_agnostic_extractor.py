# ============================================================================
# src/medical_ingestion/extractors/content_agnostic_extractor.py
# ============================================================================
"""
Content-Agnostic Medical Extractor (2-Stage Architecture)

Stage 1: Comprehensive Dump - Single LLM call extracts ALL information
Stage 2: Structuring - Parse dump into typed schema fields

Key improvements:
- Single LLM call instead of 5 parallel calls (faster, more consistent)
- Real confidence scores based on field completeness
- Document delimiters to prevent prompt injection
- Hints disabled by default (can be enabled via config)
- Honest naming: _extract_basic_patterns() for regex fallback

Inspired by Unstract's approach but optimized for medical documents.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from datetime import datetime

if TYPE_CHECKING:
    from ..core.prompt_manager import PromptManager
    from ..core.adaptive_retrieval import AdaptiveRetrieval

logger = logging.getLogger(__name__)


@dataclass
class PatientInfo:
    """Patient demographic information."""
    name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    mrn: Optional[str] = None  # Medical Record Number
    address: Optional[str] = None
    phone: Optional[str] = None
    insurance: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    """
    Generic test/measurement result.

    Works for: lab values, vital signs, imaging measurements, etc.
    """
    name: str
    value: Any  # Can be numeric or text
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    abnormal_flag: Optional[str] = None  # H, L, HIGH, LOW, CRITICAL, etc.
    category: Optional[str] = None  # "lab", "vital", "imaging_measurement"
    confidence: float = 0.0
    source_text: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None

    # Enrichment fields (added later by LabEnricher)
    loinc_code: Optional[str] = None
    loinc_name: Optional[str] = None  # Canonical LOINC component name
    validation_status: Optional[str] = None  # verified, ocr_corrected, unverified

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuple to list for JSON serialization
        if d.get('bbox'):
            d['bbox'] = list(d['bbox'])
        return d


@dataclass
class MedicationInfo:
    """Medication information from any source."""
    name: str
    strength: Optional[str] = None
    route: Optional[str] = None  # oral, topical, IV, etc.
    frequency: Optional[str] = None
    quantity: Optional[str] = None
    refills: Optional[int] = None
    instructions: Optional[str] = None
    prescriber: Optional[str] = None
    status: Optional[str] = None  # current, discontinued, new
    confidence: float = 0.0
    source_text: str = ""

    # Enrichment fields (added later by PrescriptionEnricher)
    rxcui: Optional[str] = None
    rxnorm_name: Optional[str] = None
    validation_status: Optional[str] = None  # verified, ocr_corrected, medgemma_verified, strength_mismatch, unverified

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalFinding:
    """Clinical finding, impression, or diagnosis."""
    finding: str
    category: Optional[str] = None  # "diagnosis", "impression", "recommendation", "assessment"
    severity: Optional[str] = None  # "normal", "abnormal", "critical"
    location: Optional[str] = None  # Body part/system
    confidence: float = 0.0
    source_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcedureInfo:
    """Procedure information."""
    name: str
    date: Optional[str] = None
    provider: Optional[str] = None
    findings: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DateInfo:
    """Date extracted from document."""
    date_type: str  # "collection", "report", "service", "birth"
    date_value: str
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderInfo:
    """Healthcare provider information."""
    name: str
    role: Optional[str] = None  # "physician", "nurse", "technician"
    npi: Optional[str] = None
    specialty: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrganizationInfo:
    """Healthcare organization information."""
    name: str
    address: Optional[str] = None
    phone: Optional[str] = None
    fax: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenericMedicalExtraction:
    """
    Type-agnostic extraction result.

    This is the universal output format regardless of document type.
    """
    # Patient information
    patient: Optional[PatientInfo] = None

    # Clinical content (works for ANY medical document)
    test_results: List[TestResult] = field(default_factory=list)
    medications: List[MedicationInfo] = field(default_factory=list)
    findings: List[ClinicalFinding] = field(default_factory=list)
    procedures: List[ProcedureInfo] = field(default_factory=list)

    # Dates and providers
    dates: List[DateInfo] = field(default_factory=list)
    providers: List[ProviderInfo] = field(default_factory=list)
    organizations: List[OrganizationInfo] = field(default_factory=list)

    # RAW FIELDS - Captures key-value pairs via regex (not LLM)
    # This is a best-effort fallback, not comprehensive
    raw_fields: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    extraction_confidence: float = 0.0
    extraction_method: str = "content_agnostic_v2"
    raw_llm_responses: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient": self.patient.to_dict() if self.patient else None,
            "test_results": [t.to_dict() for t in self.test_results],
            "medications": [m.to_dict() for m in self.medications],
            "findings": [f.to_dict() for f in self.findings],
            "procedures": [p.to_dict() for p in self.procedures],
            "dates": [d.to_dict() for d in self.dates],
            "providers": [p.to_dict() for p in self.providers],
            "organizations": [o.to_dict() for o in self.organizations],
            "raw_fields": self.raw_fields,
            "extraction_confidence": self.extraction_confidence,
            "extraction_method": self.extraction_method,
            "warnings": self.warnings
        }


# =============================================================================
# Dual Extraction Prompts (Language-Agnostic)
# =============================================================================
# Pass 1: Comprehensive extraction
# Pass 2: Verification of ambiguous fields

EXTRACTION_PROMPT = """Extract medical data from this document as JSON.

<DOC>
{text}
</DOC>

Return JSON with these fields. Extract ONLY actual values from the document.

{{
  "patient": {{"name": "full name", "dob": "birth date only", "gender": "M/F", "mrn": "medical record #", "address": "street address", "insurance_id": "insurance/member ID"}},
  "test_results": [{{"name": "test name", "value": "result", "unit": "unit", "reference_range": "range", "abnormal_flag": "H/L/null"}}],
  "medications": [{{"name": "drug name + strength", "frequency": "dosing"}}],
  "findings": [{{"finding": "diagnosis or impression"}}],
  "dates": [{{"date_type": "service/invoice/collection", "date_value": "the date"}}],
  "providers": [{{"name": "provider name", "role": "role/title", "license": "license #"}}],
  "organizations": [{{"name": "organization", "address": "address", "phone": "phone"}}],
  "financial": {{"invoice_number": "invoice/receipt #", "total": "total amount", "currency": "$/€/etc"}}
}}

Rules:
- Return null or [] if data doesn't exist - don't invent values
- CRITICAL: Extract EVERY test result - each row with a test name and value
- For lab reports: include ALL components (WBC, RBC, Hemoglobin, Hematocrit, Platelets,
  Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils, chemistry values, etc.)
- Include QUALITATIVE results (Negative, Positive, Trace, Few, Many, etc.) not just numeric values
- Include ALL urinalysis results: Color, Appearance, Specific Gravity, pH, Protein, Glucose,
  Ketones, Blood, Bilirubin, Urobilinogen, Nitrite, Leukocyte Esterase, Bacteria, WBC, RBC, etc.
- Include percentage results (e.g., Neutrophils 62%, Lymphocytes 25%)
- If a test shows "DNR", "Did Not Report", "NOT REPORTED", "Not Performed", "TNP", or "QNS",
  set value to the exact text shown (e.g. "DNR") — do NOT convert to "0" or any number
- For dob: only extract if it's clearly labeled as birth date/DOB/date of birth
- Separate insurance_id from address - they are different fields
- Do NOT include duplicate entries — if the same test/medication appears multiple times, include it ONLY ONCE

Return ONLY valid JSON:"""

VERIFICATION_PROMPT = """Verify these extracted values from a medical document.

Original text excerpt:
<DOC>
{text_excerpt}
</DOC>

Extracted values to verify:
{extracted_values}

For each field, answer: Is this value correct for this field type?

Return JSON:
{{
  "patient_name": {{"value": "...", "correct": true/false, "reason": "why"}},
  "patient_dob": {{"value": "...", "correct": true/false, "reason": "why - is this actually a birth date?"}},
  "patient_address": {{"value": "...", "correct": true/false, "reason": "why"}},
  "invoice_number": {{"value": "...", "correct": true/false, "reason": "why - is this a numeric ID, not a name?"}},
  "total_amount": {{"value": "...", "correct": true/false, "reason": "why - is this a currency amount?"}}
}}

Be strict:
- DOB must be explicitly labeled as birth date, not service dates or invoice dates
- Insurance IDs (like "244627-01") are NOT birth dates
- Street numbers (like "101-6545") are part of addresses, NOT birth dates
- Business names are NOT invoice numbers

Return ONLY valid JSON:"""


class ContentAgnosticExtractor:
    """
    Generic medical extraction using 2-stage architecture.

    Stage 1: Single comprehensive LLM call extracts all information
    Stage 2: Parse and structure into typed schema fields

    Key features:
    - Single LLM call (faster, more consistent than 5 parallel calls)
    - Real confidence based on field completeness
    - Document delimiters prevent prompt injection
    - Hints disabled by default (enable via use_extraction_hints=True)

    Usage:
        extractor = ContentAgnosticExtractor()
        result = await extractor.extract(text, layout)
        print(result.test_results)
        print(result.medications)
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        prompt_manager: 'PromptManager' = None,
        retrieval: 'AdaptiveRetrieval' = None
    ):
        self.config = config or {}
        self._llm_client = None
        self._prompt_manager = prompt_manager
        self._retrieval = retrieval

        # Configuration
        self.model = self.config.get('model', 'medgemma:4b')
        self.temperature = self.config.get('temperature', 0.1)
        # Output token limit for single-call extractions (prescriptions, radiology, etc).
        # The global MAX_TOKENS (.env) is typically 1000 (fine for classification),
        # but extraction JSON needs more. 2000 is enough for most single documents.
        # Lab chunked extraction uses its own higher limit (4000) in _extract_chunked().
        global_max = self.config.get('max_tokens', 2000)
        self.max_tokens = max(global_max, 2000)

        # Hints disabled by default - can cause issues and adds complexity
        self.use_hints = self.config.get('use_extraction_hints', False)

    @property
    def prompt_manager(self) -> 'PromptManager':
        """Lazy load prompt manager."""
        if self._prompt_manager is None:
            from ..core.prompt_manager import get_prompt_manager
            self._prompt_manager = get_prompt_manager(self.config)
        return self._prompt_manager

    @property
    def retrieval(self) -> Optional['AdaptiveRetrieval']:
        """Lazy load adaptive retrieval."""
        if self._retrieval is None and self.use_hints:
            try:
                from ..core.adaptive_retrieval import AdaptiveRetrieval
                self._retrieval = AdaptiveRetrieval(self.config)
            except Exception as e:
                logger.warning(f"Could not initialize AdaptiveRetrieval: {e}")
        return self._retrieval

    async def extract(
        self,
        text: str,
        layout: Optional[Any] = None,
        retrieval_strategy: str = "simple",
        extraction_hints: Dict[str, Any] = None
    ) -> GenericMedicalExtraction:
        """
        Extract all medical content using 2-stage architecture.

        Stage 1: Single LLM call for comprehensive extraction
        Stage 2: Parse response into typed schema fields

        Args:
            text: Extracted text from document
            layout: Optional layout info (from UniversalTextExtractor)
            retrieval_strategy: Strategy for context selection (only used if hints enabled)
            extraction_hints: Optional hints from similar documents

        Returns:
            GenericMedicalExtraction with all extracted data
        """
        if not text or len(text.strip()) < 50:
            logger.warning("Text too short for extraction")
            return GenericMedicalExtraction(
                warnings=["Text too short for extraction"]
            )

        # Optional: Get hints from similar documents (disabled by default)
        if extraction_hints is None and self.use_hints and self.retrieval:
            try:
                from ..core.adaptive_retrieval import RetrievalStrategy
                strategy = RetrievalStrategy(retrieval_strategy)
                retrieval_result = await self.retrieval.retrieve(
                    text=text,
                    strategy=strategy,
                    extraction_prompt="Extract all medical information"
                )
                extraction_hints = retrieval_result.extraction_hints
                logger.info(f"Got extraction hints from {len(retrieval_result.similar_docs)} similar docs")
            except Exception as e:
                logger.debug(f"Could not get extraction hints: {e}")
                extraction_hints = {}

        # For documents with many test results, use chunked extraction
        # to ensure ALL results are captured (medgemma-4b struggles with long docs)
        # Using 2500 chars per chunk for reliable extraction with local models
        max_text_length = self.config.get('max_text_length', 4000)
        logger.info(f"Extraction config: max_text_length={max_text_length}, max_tokens={self.max_tokens}")

        # Normalize text: collapse excessive whitespace from layout-preserved extraction
        # This reduces token count and prevents LLM timeouts from bloated prompts
        # Keep newlines for structure but collapse multiple spaces
        normalized_text = '\n'.join(' '.join(line.split()) for line in text.split('\n'))
        normalized_text = '\n'.join(line for line in normalized_text.split('\n') if line.strip())

        # Remove PDF layout artifacts: vertical sidebar text (REPORT, SAMPLE, etc.)
        # shows up as isolated single-letter lines or inline noise after collapsing whitespace
        # Common sidebar letters: T, R, O, P, E, S, A, M, L (from REPORT, SAMPLE)
        cleaned_lines = []
        for line in normalized_text.split('\n'):
            stripped = line.strip()
            # Remove lines that are just 1-2 characters (sidebar artifacts)
            if len(stripped) <= 2 and stripped.isalpha():
                continue
            # Remove trailing single letter appended to lab IDs: "% T01" → "% 01"
            line = re.sub(r'\s+([A-Z])(\d{2})$', r' \2', line)
            line = re.sub(r'(\d{2})\s+[A-Z]$', r'\1', line)
            # Remove inline single-letter artifacts between digits (sidebar bleed):
            # "4E.9" → "4.9", "7E5" → "75", "2O5" → "25", "1E.9" → "1.9"
            # Also before dashes: "1.9O-3.7" → "1.9-3.7"
            # These are NOT OCR misreads but inserted sidebar characters
            line = re.sub(r'(\d)[TROPESAM](\d)', r'\1\2', line)
            line = re.sub(r'(\d)[TROPESAM](\.)', r'\1\2', line)
            line = re.sub(r'(\d)[TROPESAM]([-–])', r'\1\2', line)
            # Remove leading single letter before numbers: "L177" → "177", "P0.70" → "0.70"
            # Preserve valid prefixes: T3, T4, B12 (letter + single digit = likely test name)
            # Only strip when followed by 2+ digits or digit+period (numeric value, not test name)
            line = re.sub(r'(?<=\s)([TROPESLAM])(\d{2,})', lambda m: m.group(0)
                          if m.group(1) + m.group(2)[:1] in ('T3', 'T4', 'B1', 'D2', 'D3')
                          else m.group(2), line)
            line = re.sub(r'(?<=\s)([TROPESLAM])(\d\.)', lambda m: m.group(0)
                          if m.group(1) + m.group(2)[0] in ('T3', 'T4', 'B1', 'D2', 'D3')
                          else m.group(2), line)
            cleaned_lines.append(line)
        normalized_text = '\n'.join(cleaned_lines)

        effective_length = len(normalized_text)
        logger.debug(f"Text length: raw={len(text)}, normalized={effective_length}")

        # Use normalized text for all processing
        text = normalized_text

        # Check if this looks like a lab/test document with many results
        # Match patterns: mg/dL, x10E3/uL, g/dL, %, mmol/L, U/L, etc.
        unit_patterns = len(re.findall(
            r'\d+\.?\d*\s*(?:mg/dL|g/dL|x10E\d/uL|mmol/L|U/L|ng/L|%|ratio|mL|IU|fL|pg|ug/dL)',
            text, re.IGNORECASE
        ))
        # Also check for reference range patterns
        ref_patterns = len(re.findall(r'\d+\.?\d*\s*[-–]\s*\d+\.?\d*', text))
        # Check for flag patterns (H, L markers)
        flag_patterns = len(re.findall(r'\b[HL]\b|\bHIGH\b|\bLOW\b', text))

        # Also check for urinalysis patterns (qualitative results without standard units)
        urinalysis_patterns = len(re.findall(
            r'\b(negative|positive|trace|moderate|large|few|many|rare|'
            r'clear|turbid|yellow|amber|specific gravity|ph\b|'
            r'leukocyte|nitrite|ketone|bilirubin|urobilinogen|'
            r'bacteria|epithelial|cast|crystal)\b',
            text, re.IGNORECASE
        ))

        # Lab documents: many unit patterns, reference ranges, flags, or urinalysis terms
        has_many_results = (unit_patterns > 10) or (ref_patterns > 15) or (flag_patterns > 3) or (urinalysis_patterns > 5)
        logger.debug(f"Lab detection: units={unit_patterns}, refs={ref_patterns}, flags={flag_patterns}, urinalysis={urinalysis_patterns}, is_lab={has_many_results}")

        # For lab documents: strip clinical commentary blocks before chunking.
        # Quest/LabCorp reports have multi-line notes after certain results (LDL
        # commentary, Vitamin D deficiency thresholds, HbA1c diabetes info, etc.).
        # These waste chunk space and overwhelm local 4B models — the LLM extracts
        # the first ~10 results from a 25-result chunk then stops.
        # Stripping notes can reduce 7700 chars → ~3500, cutting chunks in half.
        if has_many_results and effective_length > max_text_length:
            text_for_chunking = self._strip_lab_commentary(text)
            stripped_length = len(text_for_chunking)
            if stripped_length < effective_length:
                logger.info(f"Stripped lab commentary: {effective_length} → {stripped_length} chars ({effective_length - stripped_length} removed)")
                text = text_for_chunking
                effective_length = stripped_length

        # Use chunked extraction for long documents to avoid losing content.
        # Previously only lab docs got chunked — non-lab docs were hard-truncated
        # at max_text_length, silently losing everything after the cutoff.
        if has_many_results and effective_length > max_text_length:
            # Lab documents: use 1500-char chunks so each chunk has ≤15 test lines.
            # MedGemma 4B generates ~10-15 JSON results then closes the array,
            # so 2500-char chunks (25+ test lines) lose half the results.
            # 1500 chars ≈ 12-15 test lines = within the model's output capacity.
            chunk_size = 1500
            logger.info(f"Lab document ({effective_length} chars) with {unit_patterns} unit patterns - using chunked extraction (chunk_size={chunk_size})")
            llm_result = await self._extract_chunked(text, chunk_size, extraction_hints)
        elif effective_length > max_text_length:
            # Non-lab long document — also use chunked extraction instead of
            # hard truncation, so we don't silently lose content past the cutoff
            chunk_size = max_text_length
            logger.info(f"Long document ({effective_length} chars) exceeds {max_text_length} limit - using chunked extraction (chunk_size={chunk_size})")
            llm_result = await self._extract_chunked(text, chunk_size, extraction_hints)
        else:
            # Short document — single comprehensive extraction call
            logger.info("Pass 1: Running comprehensive extraction...")
            llm_result = await self._extract_comprehensive(text, extraction_hints)

        # PASS 2: Structure the response
        logger.info("Pass 2: Structuring extraction results...")
        extraction = self._structure_extraction(llm_result)

        # Extract invoice/financial data from LLM result BEFORE verification
        financial = llm_result.get('financial', {})
        if isinstance(financial, dict):
            if financial.get('invoice_number'):
                extraction.raw_fields['invoice_number'] = financial['invoice_number']
            if financial.get('total'):
                extraction.raw_fields['total_amount'] = financial['total']
            if financial.get('currency'):
                extraction.raw_fields['currency'] = financial['currency']

        # PASS 3: Verify ambiguous fields (optional but recommended)
        # This happens AFTER adding financial data so verification can remove bad values
        use_verification = self.config.get('use_verification', True)
        if use_verification and (extraction.patient or extraction.raw_fields.get('invoice_number')):
            logger.info("Pass 3: Verifying ambiguous fields...")
            extraction = await self._verify_and_reconcile(text, extraction, llm_result)

        # Add universal patterns (ISO dates, emails, phones - language agnostic)
        universal_fields = self._extract_universal_patterns(text)
        for key, value in universal_fields.items():
            if key not in extraction.raw_fields:
                extraction.raw_fields[key] = value

        # Calculate real confidence based on field completeness AND coverage
        extraction.extraction_confidence = self._calculate_real_confidence(extraction, text)

        return extraction

    async def _extract_comprehensive(
        self,
        text: str,
        hints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Pass 1: Single comprehensive LLM call (language-agnostic).

        Returns parsed JSON dict or empty dict on failure.
        """
        # Build prompt with document delimiters (prevents injection)
        prompt = EXTRACTION_PROMPT.format(text=text)

        # Optionally add hints if enabled and available
        if hints and hints.get('field_examples'):
            hint_text = self._format_hints_for_prompt(hints)
            if hint_text:
                prompt = f"[Context from similar documents]\n{hint_text}\n\n{prompt}"

        # Call LLM (retries internally on empty response)
        response = await self._call_llm(prompt)

        if not response:
            logger.warning(f"LLM returned empty response for comprehensive extraction (text_len={len(text)})")
            return {}

        # Parse JSON response
        parsed = self._parse_json(response)
        if not parsed:
            logger.warning("Could not parse LLM response as JSON")
            return {}

        return parsed

    async def _extract_chunked(
        self,
        text: str,
        chunk_size: int,
        hints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract from long documents in chunks, merging test results.

        This ensures we capture ALL lab values even in very long documents
        where truncation would miss results at the end.

        Strategy:
        1. First chunk: Extract patient info, providers, dates (header data)
        2. All chunks: Extract test_results
        3. Merge: Combine test_results from all chunks, deduplicate
        """
        # Calculate chunks with overlap to avoid cutting mid-result
        overlap = 500
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
            if start >= len(text) - overlap:
                break

        logger.info(f"Processing {len(chunks)} chunks for comprehensive extraction")

        # Test-only prompt for extracting lab results from chunks (faster than full prompt)
        TEST_ONLY_PROMPT = """Extract EVERY test result from this lab report section as JSON.

<DOC>
{text}
</DOC>

IMPORTANT: Extract EVERY SINGLE test result, including:
- Numeric results (e.g., WBC: 5.0 x10E3/uL)
- Qualitative results (e.g., Protein: Negative, Bacteria: Few, Color: Yellow)
- Percentage results (e.g., Neutrophils: 62%)
This includes CBC differentials, absolute counts, chemistry, urinalysis, and ALL other tests.

If a test shows "DNR", "Did Not Report", or "NOT REPORTED", set value to "DNR" — do NOT convert to "0".

Return JSON:
{{"test_results": [{{"name": "exact test name", "value": "result value", "unit": "unit if shown", "reference_range": "range if shown", "abnormal_flag": "H/L if flagged"}}]}}

Example output:
{{"test_results": [
  {{"name": "WBC", "value": "5.0", "unit": "x10E3/uL", "reference_range": "3.4-10.8", "abnormal_flag": null}},
  {{"name": "Neutrophils", "value": "55", "unit": "%", "reference_range": null, "abnormal_flag": null}},
  {{"name": "Lymphocytes", "value": "30", "unit": "%", "reference_range": null, "abnormal_flag": null}},
  {{"name": "Color", "value": "Yellow", "unit": null, "reference_range": null, "abnormal_flag": null}},
  {{"name": "Protein", "value": "Negative", "unit": null, "reference_range": "Negative", "abnormal_flag": null}},
  {{"name": "Specific Gravity", "value": "1.025", "unit": null, "reference_range": "1.005-1.030", "abnormal_flag": null}}
]}}

Extract ALL tests - do not skip any. Return [] only if truly no test values exist.

Return ONLY valid JSON:"""

        # Simpler prompt for header info (patient, provider) - use smaller text
        HEADER_PROMPT = """Extract patient and provider info from this medical document header as JSON.

<DOC>
{text}
</DOC>

Return JSON:
{{"patient": {{"name": "full name", "dob": "birth date", "gender": "M/F", "mrn": "medical record #"}},
 "providers": [{{"name": "provider name", "role": "role/title"}}],
 "organizations": [{{"name": "organization name", "address": "address", "phone": "phone"}}],
 "dates": [{{"date_type": "type", "date_value": "the date"}}]}}

Return null for missing fields. Return ONLY valid JSON:"""

        # Lab chunks can have 30+ test results, each ~50-80 tokens of JSON.
        # Use a high token limit (4000) to avoid truncated output.
        chunk_max_tokens = max(self.max_tokens, 4000)

        async def extract_tests_from_chunk(chunk_text: str) -> List[Dict]:
            prompt = TEST_ONLY_PROMPT.format(text=chunk_text)
            response = await self._call_llm(prompt, max_tokens=chunk_max_tokens)
            if not response:
                return []
            parsed = self._parse_json(response)
            if not parsed:
                return []
            raw_tests = parsed.get('test_results', [])
            # LLM sometimes returns nested lists or non-dict items — flatten and filter
            results = []
            for item in raw_tests:
                if isinstance(item, dict):
                    results.append(item)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict):
                            results.append(sub)
            return results

        async def extract_header_info(header_text: str) -> Dict[str, Any]:
            prompt = HEADER_PROMPT.format(text=header_text)
            response = await self._call_llm(prompt, max_tokens=1000)
            if not response:
                return {}
            return self._parse_json(response) or {}

        # Chunk concurrency: default 1 (sequential) for Ollama compatibility.
        # Set MAX_CONCURRENT_CHUNKS=2 in .env AND start Ollama with
        # OLLAMA_NUM_PARALLEL=2 to enable parallel chunk extraction.
        max_concurrent = self.config.get('max_concurrent_chunks', 1)

        header_text = text[:1500]

        # Extract header info first (patient, provider, dates)
        logger.debug("Extracting header info...")
        header_result = await extract_header_info(header_text)
        first_result = header_result if isinstance(header_result, dict) else {}

        if max_concurrent > 1:
            # Parallel: process multiple chunks at once (requires OLLAMA_NUM_PARALLEL)
            semaphore = asyncio.Semaphore(max_concurrent)

            async def extract_with_limit(idx: int, chunk_text: str) -> List[Dict]:
                async with semaphore:
                    logger.debug(f"Processing chunk {idx+1}/{len(chunks)}...")
                    return await extract_tests_from_chunk(chunk_text)

            chunk_tasks = [
                extract_with_limit(i, chunk) for i, chunk in enumerate(chunks)
            ]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            chunk_results = [
                r if isinstance(r, list) else [] for r in chunk_results
            ]
        else:
            # Sequential: safe default for single-threaded Ollama
            # Per-chunk timeout prevents one stuck chunk from hanging the loop
            per_chunk_timeout = self.config.get('per_chunk_timeout', 120)  # 2 min per chunk
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}...")
                try:
                    tests = await asyncio.wait_for(
                        extract_tests_from_chunk(chunk),
                        timeout=per_chunk_timeout
                    )
                    chunk_results.append(tests)
                except (asyncio.TimeoutError, TimeoutError):
                    logger.warning(
                        f"Chunk {i+1}/{len(chunks)} timed out after {per_chunk_timeout}s, skipping"
                    )
                    chunk_results.append([])
                except Exception as e:
                    logger.warning(f"Chunk {i+1}/{len(chunks)} failed: {e}")
                    chunk_results.append([])

        # Merge all test results with smart deduplication
        # Strategy: Keep entries with actual values, dedupe by name (prefer complete entries)
        all_tests: List[Dict] = []
        for chunk_tests in chunk_results:
            all_tests.extend(chunk_tests)

        # Build a dict keyed by lowercase test name
        # If we have duplicates, prefer the one with actual values
        tests_by_name = {}

        def has_value(test: Dict) -> bool:
            """Check if test has an actual (non-null, non-empty) value."""
            val = test.get('value')
            return val is not None and str(val).strip() not in ('', 'null', 'None')

        def completeness_score(test: Dict) -> int:
            """Score how complete a test result entry is."""
            score = 0
            if has_value(test):
                score += 10  # Value is most important
            if test.get('unit'):
                score += 2
            if test.get('reference_range'):
                score += 2
            if test.get('abnormal_flag'):
                score += 1
            return score

        # Process all test results from all chunks
        for test in all_tests:
            name_key = test.get('name', '').lower().strip()
            if not name_key:
                continue
            if name_key not in tests_by_name:
                tests_by_name[name_key] = test
            elif completeness_score(test) > completeness_score(tests_by_name[name_key]):
                tests_by_name[name_key] = test

        # Convert back to list, filtering out tests without actual values
        # Only keep tests that have a real value (not just reference_range)
        merged_tests = [
            test for test in tests_by_name.values()
            if has_value(test)
        ]

        first_result['test_results'] = merged_tests
        logger.info(f"Chunked extraction found {len(merged_tests)} test results with values (filtered from {len(tests_by_name)})")

        return first_result

    async def _verify_and_reconcile(
        self,
        original_text: str,
        extraction: 'GenericMedicalExtraction',
        llm_result: Dict[str, Any]
    ) -> 'GenericMedicalExtraction':
        """
        Pass 3: Verify extracted values using a second LLM call.

        This helps catch common errors like:
        - Insurance IDs mistaken for DOB
        - Street numbers mistaken for DOB
        - Business names mistaken for invoice numbers
        - Service dates mistaken for DOB
        """
        # Build values to verify
        values_to_verify = {}

        if extraction.patient:
            if extraction.patient.name:
                values_to_verify['patient_name'] = extraction.patient.name
            if extraction.patient.dob:
                values_to_verify['patient_dob'] = extraction.patient.dob
            if extraction.patient.address:
                values_to_verify['patient_address'] = extraction.patient.address

        # Get financial data from LLM result
        financial = llm_result.get('financial', {})
        if isinstance(financial, dict):
            if financial.get('invoice_number'):
                values_to_verify['invoice_number'] = financial['invoice_number']
            if financial.get('total'):
                values_to_verify['total_amount'] = financial['total']

        # Skip verification if nothing to verify
        if not values_to_verify:
            return extraction

        # Build verification prompt
        text_excerpt = original_text[:1500]  # Use first 1500 chars for context
        values_json = json.dumps(values_to_verify, indent=2)
        prompt = VERIFICATION_PROMPT.format(
            text_excerpt=text_excerpt,
            extracted_values=values_json
        )

        # Call LLM for verification — needs very few tokens (just field verdicts)
        response = await self._call_llm(prompt, max_tokens=500)
        if not response:
            logger.warning("Verification LLM call failed, keeping original extraction")
            return extraction

        # Parse verification response
        verification = self._parse_json(response)
        if not verification:
            logger.warning("Could not parse verification response")
            return extraction

        # Reconcile: null out fields that failed verification
        logger.info(f"Verification results: {verification}")

        if extraction.patient:
            # Check DOB verification
            dob_check = verification.get('patient_dob', {})
            if isinstance(dob_check, dict) and dob_check.get('correct') is False:
                reason = dob_check.get('reason', 'failed verification')
                logger.info(f"DOB '{extraction.patient.dob}' failed verification: {reason}")
                extraction.patient.dob = None
                extraction.warnings.append(f"DOB removed: {reason}")

            # Check name verification
            name_check = verification.get('patient_name', {})
            if isinstance(name_check, dict) and name_check.get('correct') is False:
                reason = name_check.get('reason', 'failed verification')
                logger.info(f"Patient name '{extraction.patient.name}' failed verification: {reason}")
                # Don't null patient name, but lower confidence
                extraction.patient.confidence = max(0.1, extraction.patient.confidence - 0.3)
                extraction.warnings.append(f"Patient name uncertain: {reason}")

        # Check invoice number
        invoice_check = verification.get('invoice_number', {})
        if isinstance(invoice_check, dict) and invoice_check.get('correct') is False:
            reason = invoice_check.get('reason', 'failed verification')
            logger.info(f"Invoice number failed verification: {reason}")
            # Remove from raw_fields if present
            if 'invoice_number' in extraction.raw_fields:
                del extraction.raw_fields['invoice_number']
            extraction.warnings.append(f"Invoice number removed: {reason}")

        return extraction

    def _structure_extraction(
        self,
        llm_result: Dict[str, Any]
    ) -> GenericMedicalExtraction:
        """
        Stage 2: Convert LLM output to typed schema.

        Maps the comprehensive dump to typed dataclasses with real confidence.
        """
        warnings = []

        if not llm_result:
            warnings.append("LLM returned empty response")
            return GenericMedicalExtraction(warnings=warnings)

        # Import OCR correction utilities
        try:
            from ..processors.lab.utils.parsing import correct_lab_value_ocr
            from ..utils.text_normalizer import fix_spurious_spaces
        except ImportError:
            correct_lab_value_ocr = None
            fix_spurious_spaces = None

        # DNR values — test was not performed, don't store as a result
        DNR_VALUES = {
            'dnr', 'did not report', 'not reported', 'not performed',
            'tnp', 'qns', 'quantity not sufficient', 'cancelled',
            'see note', 'see comment', 'pending',
        }

        # Parse patient info
        patient = None
        patient_data = llm_result.get('patient', {})
        if patient_data and isinstance(patient_data, dict):
            # Calculate confidence with tiered fields:
            # Core fields (name, dob, gender, mrn) are expected on most medical docs.
            # Bonus fields (address, phone, insurance) are rarely on lab reports
            # and shouldn't penalize the score when absent.
            core_fields = ['name', 'dob', 'gender', 'mrn']
            bonus_fields = ['address', 'phone', 'insurance', 'insurance_id']
            core_filled = sum(1 for f in core_fields if patient_data.get(f))
            bonus_filled = sum(1 for f in bonus_fields if patient_data.get(f))
            # Core fields are worth 80% of patient confidence, bonus 20%
            confidence = (core_filled / len(core_fields)) * 0.8 + (bonus_filled / len(bonus_fields)) * 0.2

            # Handle insurance_id -> insurance mapping
            insurance = patient_data.get('insurance') or patient_data.get('insurance_id')

            # Fix spurious spaces in patient text fields
            p_name = patient_data.get('name')
            p_dob = patient_data.get('dob')
            if fix_spurious_spaces:
                if p_name:
                    p_name = fix_spurious_spaces(p_name)
                if p_dob:
                    p_dob = fix_spurious_spaces(p_dob)

            patient = PatientInfo(
                name=p_name,
                dob=p_dob,
                gender=patient_data.get('gender'),
                mrn=patient_data.get('mrn'),
                address=patient_data.get('address'),
                phone=patient_data.get('phone'),
                insurance=insurance,
                confidence=round(confidence, 2)
            )

        # Parse test results - only include tests with actual values
        test_results = []
        for item in llm_result.get('test_results', []):
            if isinstance(item, dict) and item.get('name'):
                # Skip tests without actual values (only reference_range is not enough)
                value = item.get('value')
                if value is None or str(value).strip() in ('', 'null', 'None'):
                    logger.debug(f"Skipping test '{item.get('name')}' - no actual value")
                    continue

                # Skip DNR / Not Reported values
                # Also handle sidebar-corrupted DNR: "LDNR" → strip leading alpha
                value_check = str(value).strip().lower()
                value_check_cleaned = re.sub(r'^[a-z](?=dnr)', '', value_check)
                if value_check in DNR_VALUES or value_check_cleaned in DNR_VALUES:
                    logger.debug(f"Skipping DNR test '{item.get('name')}' - value: {value}")
                    continue

                # Apply OCR value corrections
                if correct_lab_value_ocr:
                    raw_value = str(value).strip()
                    corrected_value = correct_lab_value_ocr(raw_value)
                    if corrected_value != raw_value:
                        logger.debug(f"OCR value fix: '{raw_value}' → '{corrected_value}' for test '{item.get('name')}'")
                        value = corrected_value

                    ref_range = item.get('reference_range')
                    if ref_range and isinstance(ref_range, str):
                        corrected_ref = correct_lab_value_ocr(ref_range)
                        if corrected_ref != ref_range:
                            logger.debug(f"OCR ref fix: '{ref_range}' → '{corrected_ref}' for test '{item.get('name')}'")
                            item['reference_range'] = corrected_ref

                # Confidence based on completeness
                test_fields = ['value', 'unit', 'reference_range']
                filled = sum(1 for f in test_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(test_fields))  # Base 0.5 + up to 0.5 for fields

                # Normalize abnormal_flag - LLM sometimes returns string "null" instead of None
                abnormal_flag = item.get('abnormal_flag')
                if abnormal_flag and str(abnormal_flag).lower() in ('null', 'none', ''):
                    abnormal_flag = None

                test_results.append(TestResult(
                    name=item.get('name', ''),
                    value=value,
                    unit=item.get('unit'),
                    reference_range=item.get('reference_range'),
                    abnormal_flag=abnormal_flag,
                    category=item.get('category', 'other'),
                    confidence=round(confidence, 2),
                    source_text=item.get('source_text', '')
                ))

        # Deduplicate test results — LLM may return the same test twice
        # (common with multi-page docs or overlapping sections)
        if len(test_results) > 1:
            seen = {}
            deduped = []
            for t in test_results:
                key = (t.name.lower().strip(), str(t.value).strip().lower())
                if key not in seen:
                    seen[key] = t
                    deduped.append(t)
                else:
                    # Keep the one with higher confidence (more filled fields)
                    if t.confidence > seen[key].confidence:
                        deduped[deduped.index(seen[key])] = t
                        seen[key] = t
            if len(deduped) < len(test_results):
                logger.info(f"Deduplicated test results: {len(test_results)} → {len(deduped)}")
                test_results = deduped

        # Log summary of extracted tests
        logger.info(f"Structured {len(test_results)} test results from LLM output")
        if test_results:
            test_names = [t.name for t in test_results]
            logger.debug(f"Test names: {test_names}")
            # Check for common CBC differentials
            cbc_tests = ['wbc', 'rbc', 'hemoglobin', 'hematocrit', 'platelet', 'neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil', 'baso']
            found_cbc = [name for name in test_names if any(cbc.lower() in name.lower() for cbc in cbc_tests)]
            if found_cbc:
                logger.debug(f"CBC tests found: {found_cbc}")

        # Parse medications
        medications = []
        for item in llm_result.get('medications', []):
            if isinstance(item, dict) and item.get('name'):
                # Confidence based on completeness
                med_fields = ['strength', 'frequency', 'instructions']
                filled = sum(1 for f in med_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(med_fields))

                medications.append(MedicationInfo(
                    name=item.get('name', ''),
                    strength=item.get('strength'),
                    route=item.get('route'),
                    frequency=item.get('frequency'),
                    quantity=item.get('quantity'),
                    refills=item.get('refills'),
                    instructions=item.get('instructions'),
                    prescriber=item.get('prescriber'),
                    status=item.get('status'),
                    confidence=round(confidence, 2),
                    source_text=item.get('source_text', '')
                ))

        # Deduplicate medications
        if len(medications) > 1:
            seen_meds = set()
            deduped_meds = []
            for m in medications:
                key = m.name.lower().strip()
                if key not in seen_meds:
                    seen_meds.add(key)
                    deduped_meds.append(m)
            if len(deduped_meds) < len(medications):
                logger.info(f"Deduplicated medications: {len(medications)} → {len(deduped_meds)}")
                medications = deduped_meds

        # Parse findings
        findings = []
        for item in llm_result.get('findings', []):
            if isinstance(item, dict) and item.get('finding'):
                # Confidence based on completeness
                finding_fields = ['category', 'severity', 'location']
                filled = sum(1 for f in finding_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(finding_fields))

                findings.append(ClinicalFinding(
                    finding=item.get('finding', ''),
                    category=item.get('category'),
                    severity=item.get('severity'),
                    location=item.get('location'),
                    confidence=round(confidence, 2),
                    source_text=item.get('source_text', '')
                ))

        # Parse dates
        dates = []
        for item in llm_result.get('dates', []):
            if isinstance(item, dict) and item.get('date_value'):
                date_val = item.get('date_value', '')
                if fix_spurious_spaces and date_val:
                    date_val = fix_spurious_spaces(date_val)
                dates.append(DateInfo(
                    date_type=item.get('date_type', 'other'),
                    date_value=date_val,
                    confidence=0.8  # Dates are usually accurate if found
                ))

        # Parse providers
        providers = []
        for item in llm_result.get('providers', []):
            if isinstance(item, dict) and item.get('name'):
                provider_fields = ['role', 'specialty', 'npi']
                filled = sum(1 for f in provider_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(provider_fields))

                prov_name = item.get('name', '')
                if fix_spurious_spaces and prov_name:
                    prov_name = fix_spurious_spaces(prov_name)

                providers.append(ProviderInfo(
                    name=prov_name,
                    role=item.get('role'),
                    npi=item.get('npi'),
                    specialty=item.get('specialty'),
                    confidence=round(confidence, 2)
                ))

        # Parse organizations
        organizations = []
        for item in llm_result.get('organizations', []):
            if isinstance(item, dict) and item.get('name'):
                org_fields = ['address', 'phone']
                filled = sum(1 for f in org_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(org_fields))

                organizations.append(OrganizationInfo(
                    name=item.get('name', ''),
                    address=item.get('address'),
                    phone=item.get('phone'),
                    confidence=round(confidence, 2)
                ))

        # Capture additional fields from LLM
        additional = llm_result.get('additional_fields', {})
        if not isinstance(additional, dict):
            additional = {}

        # Capture invoice data into raw_fields
        invoice_data = llm_result.get('invoice', {})
        if isinstance(invoice_data, dict) and invoice_data:
            for key, value in invoice_data.items():
                if value:  # Only add non-empty values
                    field_name = f"invoice_{key}" if not key.startswith('invoice') else key
                    additional[field_name] = value

        return GenericMedicalExtraction(
            patient=patient,
            test_results=test_results,
            medications=medications,
            findings=findings,
            dates=dates,
            providers=providers,
            organizations=organizations,
            raw_fields=additional,
            raw_llm_responses={'comprehensive': str(llm_result)[:500]},
            warnings=warnings
        )

    async def _call_llm(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        """Call LLM for extraction with retry on empty response.

        Empty responses commonly happen when Ollama is swapping models
        (unloading VLM, loading MedGemma) — a brief retry usually succeeds.
        """
        if self._llm_client is None:
            from ..medgemma.client import create_client
            # Extraction needs longer timeout than classification —
            # chunked lab extraction can generate 2500+ tokens at ~30 tok/s = ~90s
            extraction_config = dict(self.config)
            extraction_config['timeout'] = max(extraction_config.get('timeout', 120), 300)
            self._llm_client = create_client(extraction_config)

        max_retries = 2
        prompt_len = len(prompt)

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"LLM call attempt {attempt}/{max_retries} (prompt: {prompt_len} chars)")
                response = await self._llm_client.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens
                )
                # Response is a dict with 'text' key from Ollama client
                text = response.get('text', '') if isinstance(response, dict) else response

                if text and text.strip():
                    return text

                # Empty response — log details and retry
                gen_tokens = response.get('generated_tokens', '?') if isinstance(response, dict) else '?'
                logger.warning(
                    f"LLM returned empty response (attempt {attempt}/{max_retries}, "
                    f"prompt_chars={prompt_len}, generated_tokens={gen_tokens})"
                )

                if attempt < max_retries:
                    # Wait before retry — gives Ollama time to finish model swap
                    wait_secs = 3 * attempt
                    logger.info(f"Retrying in {wait_secs}s (model may be loading)...")
                    await asyncio.sleep(wait_secs)

            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(3)
                else:
                    return None

        return None

    def _parse_json(self, response: str) -> Any:
        """Parse JSON from LLM response with repair."""
        if not response:
            return None

        # Try to find JSON in response
        response = response.strip()

        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Look for JSON object (the prompt asks for object, not array)
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try json_repair library
        try:
            from json_repair import repair_json
            repaired = repair_json(response)
            return json.loads(repaired)
        except Exception:
            pass

        logger.warning(f"Could not parse JSON from response: {response[:200]}")
        return None

    def _format_hints_for_prompt(self, hints: Dict[str, Any]) -> str:
        """Format extraction hints for inclusion in prompt."""
        if not hints or not hints.get('field_examples'):
            return ""

        hint_parts = []
        field_examples = hints.get('field_examples', {})

        for field_name, examples in field_examples.items():
            if examples:
                values = []
                for ex in examples[:2]:  # Max 2 examples per field
                    val = ex.get('value')
                    if val and str(val) not in values:
                        values.append(str(val)[:50])  # Truncate long values

                if values:
                    hint_parts.append(f"- {field_name}: {', '.join(values)}")

        return "\n".join(hint_parts[:10])  # Max 10 hints

    def _calculate_real_confidence(self, extraction: GenericMedicalExtraction, source_text: str = "") -> float:
        """
        Calculate real confidence based on:
        1. Quality — how complete are the extracted items (values, units, refs)?
        2. Coverage — did we extract most of what's in the document?

        Without coverage, 27 perfectly-extracted tests from an 80-test document
        would score 97% — a dangerous lie. Coverage catches this.
        """
        # === QUALITY SCORE ===
        # (score, weight) pairs — weight by number of items in the section
        weighted_scores = []

        if extraction.patient:
            weighted_scores.append((extraction.patient.confidence, 1))

        if extraction.test_results:
            avg_confidence = sum(t.confidence for t in extraction.test_results) / len(extraction.test_results)
            weight = min(len(extraction.test_results), 10)
            weighted_scores.append((avg_confidence, weight))

        if extraction.medications:
            avg_confidence = sum(m.confidence for m in extraction.medications) / len(extraction.medications)
            weight = min(len(extraction.medications), 5)
            weighted_scores.append((avg_confidence, weight))

        if extraction.findings:
            avg_confidence = sum(f.confidence for f in extraction.findings) / len(extraction.findings)
            weight = min(len(extraction.findings), 5)
            weighted_scores.append((avg_confidence, weight))

        if extraction.dates:
            weighted_scores.append((0.8, 1))

        if extraction.providers:
            avg_confidence = sum(p.confidence for p in extraction.providers) / len(extraction.providers)
            weighted_scores.append((avg_confidence, 1))

        if extraction.organizations:
            avg_confidence = sum(o.confidence for o in extraction.organizations) / len(extraction.organizations)
            weighted_scores.append((avg_confidence, 1))

        if not weighted_scores:
            return 0.2

        total_weight = sum(w for _, w in weighted_scores)
        quality_score = sum(score * w for score, w in weighted_scores) / total_weight
        section_bonus = min(0.15, len(weighted_scores) * 0.03)
        quality_score = min(1.0, quality_score + section_bonus)

        # === COVERAGE SCORE ===
        # Estimate how many items the document SHOULD have, compare to what
        # we actually extracted. Without this, extracting 3 of 10 medications
        # perfectly scores 97% — a dangerous false confidence.
        #
        # Strategy: use document-type-agnostic heuristics to count expected
        # items in the source text, then pick the best match.
        coverage_score = 1.0
        if source_text:
            coverage_score = self._estimate_coverage(source_text, extraction)

        # Final confidence = quality × coverage
        # If quality is 97% but coverage is 34% (27/80), final = 0.97 × 0.34 = 33%
        final = round(quality_score * coverage_score, 3)
        logger.info(f"Confidence: quality={quality_score:.2f} × coverage={coverage_score:.2f} = {final:.2f}")
        return final

    def _estimate_coverage(self, source_text: str, extraction: GenericMedicalExtraction) -> float:
        """Estimate what fraction of the document's content was actually extracted.

        Works across document types by counting type-specific patterns in the
        source text and comparing to extraction counts.

        Returns:
            Coverage ratio 0.0-1.0. Returns 1.0 if no reliable estimate is possible.
        """
        estimates = []

        # --- Lab tests ---
        # Count lines with NORMAL/HIGH/LOW flags (standard lab result format)
        expected_tests = len(re.findall(
            r'\b(?:NORMAL|HIGH|LOW|ABNORMAL)\b', source_text
        ))
        if expected_tests > 5 and extraction.test_results:
            extracted = len(extraction.test_results)
            ratio = min(1.0, extracted / expected_tests)
            estimates.append(('tests', extracted, expected_tests, ratio))

        # --- Medications ---
        # Count lines with drug dosage/form patterns:
        # "TAB.", "CAP.", "mg", "ml", "mcg", "1-0-1", "BD", "TID", "OD"
        expected_meds = len(re.findall(
            r'(?:'
            r'\b(?:TAB|CAP|INJ|SYR|SUSP|OINT|CREAM|GEL|DROP)\b\.?'
            r'|\b\d+\s*(?:mg|ml|mcg|iu|units)\b'
            r'|\b(?:once|twice|thrice)\s+(?:daily|a day)\b'
            r'|\b[1-3]-[0-1]-[0-1]-?[0-1]?\b'
            r'|\b(?:OD|BD|TDS|TID|QID|QHS|PRN|SOS|HS)\b'
            r')',
            source_text, re.IGNORECASE
        ))
        if expected_meds > 2 and extraction.medications:
            extracted = len(extraction.medications)
            ratio = min(1.0, extracted / expected_meds)
            estimates.append(('medications', extracted, expected_meds, ratio))

        # --- Findings/diagnoses ---
        # Count ICD-like codes or numbered finding patterns
        expected_findings = len(re.findall(
            r'(?:'
            r'[A-Z]\d{2}\.?\d*'  # ICD-10 codes (e.g. E11.9, J06.9)
            r'|\b(?:IMPRESSION|FINDING|DIAGNOSIS|ASSESSMENT)\s*[:\d]'
            r')',
            source_text
        ))
        if expected_findings > 2 and extraction.findings:
            extracted = len(extraction.findings)
            ratio = min(1.0, extracted / expected_findings)
            estimates.append(('findings', extracted, expected_findings, ratio))

        if not estimates:
            return 1.0  # No reliable estimate — don't penalize

        # Use the estimate with the most expected items (most reliable signal)
        best = max(estimates, key=lambda e: e[2])
        doc_type, extracted, expected, ratio = best

        logger.info(
            f"Coverage ({doc_type}): {extracted} extracted / {expected} expected "
            f"= {ratio:.0%}"
        )

        return ratio

    @staticmethod
    def _strip_lab_commentary(text: str) -> str:
        """Strip multi-line clinical commentary from lab text before chunking.

        Lab reports (Quest, LabCorp) have verbose notes after certain results:
        - LDL-C calculation method references
        - Vitamin D deficiency thresholds
        - HbA1c diabetes interpretation
        - eGFR African-American notes
        - Creatinine age-adjustment notes

        These waste chunk space and overwhelm small LLMs. A 4-page Quest report
        drops from ~7700 to ~3500 chars, halving the chunks needed.

        Strategy: keep lines that look like test results or section headers,
        drop consecutive non-result lines (commentary blocks).
        """
        result_pattern = re.compile(
            r'(?:'
            # Lines with NORMAL/HIGH/LOW/ABNORMAL flags (standard lab result format)
            r'\b(?:NORMAL|HIGH|LOW|ABNORMAL)\b'
            r'|'
            # Section headers in ALL CAPS (CBC, LIPID PANEL, URINALYSIS, etc.)
            r'^[A-Z][A-Z /,\(\)]{5,}$'
            r'|'
            # Lines with standard lab units
            r'\b(?:mg/dL|g/dL|mmol/L|mIU/L|mcg/dL|ng/mL|U/L|cells/uL|Thousand/uL|Million/uL|fL|pg|mm/h|mg/L|/HPF|/LPF|% of total)\b'
            r'|'
            # Lines with DNR, NONE SEEN, NEGATIVE, POSITIVE (qualitative results)
            r'\b(?:DNR|NONE SEEN|NEGATIVE|POSITIVE|TRACE)\b'
            r'|'
            # Page number lines (keep for structure)
            r'^\d+ of \d+$'
            r'|'
            # Patient/specimen header lines
            r'\b(?:PATIENT|SPECIMEN|DOB|GENDER|COLLECTED|RECEIVED|REPORTED|REQUISITION|ORDERING)\b'
            r'|'
            # Performing lab info
            r'\b(?:Performing Laboratory|Medical Director|Quest Diagnostics|LabCorp)\b'
            r')',
            re.IGNORECASE
        )

        lines = text.split('\n')
        kept = []
        consecutive_skips = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if result_pattern.search(stripped):
                kept.append(line)
                consecutive_skips = 0
            else:
                consecutive_skips += 1
                # Keep isolated non-result lines (might be test names without flags)
                # but drop blocks of 2+ consecutive commentary lines
                if consecutive_skips <= 1:
                    kept.append(line)

        return '\n'.join(kept)

    def _extract_universal_patterns(self, text: str) -> Dict[str, Any]:
        """
        Extract only truly universal patterns (language-agnostic).

        These patterns work regardless of document language:
        - ISO dates (YYYY-MM-DD)
        - Email addresses
        - International phone formats

        All other extraction is handled by the LLM which is language-agnostic.
        """
        raw_fields = {}

        if not text:
            return raw_fields

        # ISO dates (universal format: 2025-02-26)
        iso_dates = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
        if iso_dates:
            # Store unique dates
            unique_dates = list(dict.fromkeys(iso_dates))[:3]
            for i, dt in enumerate(unique_dates):
                key = f'date_iso_{i+1}' if i > 0 else 'date_iso'
                raw_fields[key] = dt

        # Email addresses (universal format)
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w{2,}', text)
        if emails:
            raw_fields['email'] = emails[0]

        # Phone numbers (flexible international format)
        # Matches: (450) 672-9991, 450-672-9991, +1 450 672 9991, etc.
        phones = re.findall(r'[\+]?[\d\s\-\.\(\)]{10,}', text)
        for phone in phones:
            # Clean and validate (must have at least 10 digits)
            digits = re.sub(r'\D', '', phone)
            if 10 <= len(digits) <= 15:
                raw_fields['phone'] = phone.strip()
                break

        logger.debug(f"Extracted {len(raw_fields)} universal pattern fields")
        return raw_fields


# Convenience function
async def extract_medical_content(
    text: str,
    config: Dict[str, Any] = None
) -> GenericMedicalExtraction:
    """
    Convenience function for content-agnostic medical extraction.

    Args:
        text: Document text to extract from
        config: Optional configuration

    Returns:
        GenericMedicalExtraction with all extracted data
    """
    extractor = ContentAgnosticExtractor(config or {})
    return await extractor.extract(text)
