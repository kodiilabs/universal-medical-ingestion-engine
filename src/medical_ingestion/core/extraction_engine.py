# ============================================================================
# src/medical_ingestion/core/extraction_engine.py
# ============================================================================
"""
Multi-Method Extraction Engine with Consensus Merge

Inspired by Unstract's approach:
1. Run multiple extraction methods in parallel
2. Merge results using consensus logic
3. Assign field-level confidence based on agreement

Methods:
- Template matching (highest accuracy on known formats)
- Table extraction (Camelot/pdfplumber)
- Vector DB guided (finds similar past extractions)
- MedGemma LLM (general fallback)

Consensus Logic:
- All agree → HIGH confidence (0.98)
- 2+ agree → MEDIUM confidence (0.90)
- 1 agrees → LOW confidence (0.75)
- None agree → VERY LOW, flag for review (0.50)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio
import logging
from enum import Enum
from collections import Counter
import numpy as np

from .context.extracted_value import ExtractedValue
from .config import get_config

# Import document pipeline components
from .document_pipeline import DocumentPipeline, PipelineResult


class AgreementLevel(Enum):
    """How many extraction methods agreed on a value."""
    UNANIMOUS = "unanimous"      # All methods agree
    MAJORITY = "majority"        # 2+ of 3+ agree
    MINORITY = "minority"        # Only 1 method found it
    CONFLICT = "conflict"        # Methods disagree
    NONE = "none"               # No method found it


@dataclass
class MethodResult:
    """Result from a single extraction method."""
    method_name: str
    values: Dict[str, Any]  # field_name -> value
    units: Dict[str, str]   # field_name -> unit
    confidence: float       # Method's overall confidence
    extraction_time: float = 0.0
    error: Optional[str] = None

    def get_value(self, field_name: str) -> Optional[Any]:
        """Get extracted value for a field, normalized."""
        return self.values.get(field_name)

    def has_field(self, field_name: str) -> bool:
        return field_name in self.values and self.values[field_name] is not None


@dataclass
class ConsensusField:
    """Consensus result for a single field."""
    field_name: str
    final_value: Any
    final_unit: Optional[str]
    confidence: float
    agreement_level: AgreementLevel

    # Which methods contributed
    contributing_methods: List[str] = field(default_factory=list)
    method_values: Dict[str, Any] = field(default_factory=dict)

    # Flags
    needs_review: bool = False
    review_reason: Optional[str] = None


@dataclass
class ConsensusResult:
    """Complete consensus extraction result."""
    fields: Dict[str, ConsensusField] = field(default_factory=dict)
    method_results: Dict[str, MethodResult] = field(default_factory=dict)

    overall_confidence: float = 0.0
    fields_extracted: int = 0
    fields_high_confidence: int = 0
    fields_need_review: int = 0

    extraction_time: float = 0.0

    def add_field(self, consensus_field: ConsensusField):
        self.fields[consensus_field.field_name] = consensus_field
        self.fields_extracted += 1

        if consensus_field.confidence >= 0.90:
            self.fields_high_confidence += 1
        if consensus_field.needs_review:
            self.fields_need_review += 1

    def calculate_overall_confidence(self):
        """Calculate overall confidence from field confidences."""
        if not self.fields:
            self.overall_confidence = 0.0
            return

        confidences = [f.confidence for f in self.fields.values()]
        self.overall_confidence = sum(confidences) / len(confidences)

    def to_extracted_values(self) -> List[ExtractedValue]:
        """Convert to list of ExtractedValue for ProcessingContext."""
        extracted = []
        for name, cf in self.fields.items():
            ev = ExtractedValue(
                field_name=name,
                value=cf.final_value,
                unit=cf.final_unit,
                confidence=cf.confidence,
                extraction_method=f"consensus_{cf.agreement_level.value}",
            )
            if cf.needs_review:
                ev.warnings.append(cf.review_reason or "Low agreement between methods")
            extracted.append(ev)
        return extracted


class ExtractionEngine:
    """
    Multi-method extraction with consensus merge.

    Runs template, table, vector DB, and LLM extraction in parallel,
    then merges results based on agreement.
    """

    # Base confidence scores based on agreement level
    # These are starting points - actual confidence is calculated dynamically
    CONFIDENCE_LEVELS = {
        AgreementLevel.UNANIMOUS: 0.98,
        AgreementLevel.MAJORITY: 0.92,
        AgreementLevel.MINORITY: 0.75,  # Base, can be boosted by validation
        AgreementLevel.CONFLICT: 0.50,
        AgreementLevel.NONE: 0.0,
    }

    # Expected ranges for common lab tests (for confidence boosting)
    # Format: test_name_pattern -> (unit_patterns, min_value, max_value)
    EXPECTED_RANGES = {
        "wbc": (["%", "x10"], 3.0, 15.0),
        "rbc": (["x10", "m/ul"], 3.5, 6.5),
        "hemoglobin": (["g/dl", "g/l"], 10.0, 20.0),
        "hgb": (["g/dl", "g/l"], 10.0, 20.0),
        "hematocrit": (["%"], 30.0, 60.0),
        "hct": (["%"], 30.0, 60.0),
        "mcv": (["fl"], 70.0, 110.0),
        "mch": (["pg"], 24.0, 36.0),
        "mchc": (["g/dl", "%"], 30.0, 38.0),
        "platelets": (["x10", "k/ul"], 100.0, 500.0),
        "plt": (["x10", "k/ul"], 100.0, 500.0),
        "neutrophils": (["%"], 35.0, 80.0),
        "lymphocytes": (["%"], 15.0, 50.0),
        "monocytes": (["%"], 0.0, 15.0),
        "eosinophils": (["%"], 0.0, 10.0),
        "basophils": (["%"], 0.0, 3.0),
        "glucose": (["mg/dl", "mmol"], 50.0, 200.0),
        "bun": (["mg/dl"], 5.0, 40.0),
        "creatinine": (["mg/dl"], 0.3, 3.0),
        "sodium": (["meq", "mmol"], 130.0, 150.0),
        "potassium": (["meq", "mmol"], 3.0, 6.0),
        "chloride": (["meq", "mmol"], 90.0, 115.0),
        "ast": (["u/l", "iu"], 5.0, 100.0),
        "alt": (["u/l", "iu"], 5.0, 100.0),
        "bilirubin": (["mg/dl"], 0.0, 3.0),
        "albumin": (["g/dl"], 2.5, 6.0),
        "tsh": (["miu", "uiu"], 0.1, 10.0),
        "hba1c": (["%"], 3.0, 15.0),
        "a1c": (["%"], 3.0, 15.0),
    }

    def __init__(self, config: Dict[str, Any] = None):
        # Merge env config with passed config (passed config takes precedence)
        env_config = get_config()
        self.config = {**env_config, **(config or {})}
        self.logger = logging.getLogger(__name__)

        # Initialize extractors lazily
        self._template_extractor = None
        self._table_extractor = None
        self._vector_store = None
        self._llm_client = None
        self._paddle_ocr = None
        self._document_pipeline = None

        # Pipeline configuration
        self.enable_preprocessing = self.config.get('enable_preprocessing', True)
        self.enable_region_detection = self.config.get('enable_region_detection', True)
        self.use_vlm_classification = self.config.get('use_vlm_classification', False)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup all clients."""
        await self.close()
        return False

    async def close(self):
        """Close all client sessions to prevent resource leaks."""
        if self._paddle_ocr is not None:
            try:
                await self._paddle_ocr.close()
            except Exception as e:
                self.logger.debug(f"Error closing PaddleOCR-VL client: {e}")

        if self._llm_client is not None:
            try:
                if hasattr(self._llm_client, 'close'):
                    await self._llm_client.close()
            except Exception as e:
                self.logger.debug(f"Error closing LLM client: {e}")

        self.logger.debug("ExtractionEngine clients closed")

    @property
    def template_extractor(self):
        if self._template_extractor is None:
            from medical_ingestion.processors.lab.utils.template_loader import load_template
            self._template_extractor = load_template
        return self._template_extractor

    @property
    def table_extractor(self):
        if self._table_extractor is None:
            from ..extractors.table_extractor import TableExtractor
            self._table_extractor = TableExtractor(
                use_ocr_fallback=self.config.get('use_ocr', True),
                use_vision_fallback=self.config.get('use_vision', True)
            )
        return self._table_extractor

    @property
    def vector_store(self):
        if self._vector_store is None:
            from .vector_store import VectorStore
            self._vector_store = VectorStore(self.config)
        return self._vector_store

    @property
    def llm_client(self):
        if self._llm_client is None:
            from ..medgemma.client import create_client
            self._llm_client = create_client(self.config)
        return self._llm_client

    @property
    def paddle_ocr(self):
        if self._paddle_ocr is None:
            from ..extractors.paddle_ocr import get_paddle_ocr_extractor
            self._paddle_ocr = get_paddle_ocr_extractor(self.config)
        return self._paddle_ocr

    @property
    def document_pipeline(self) -> DocumentPipeline:
        """
        Document preprocessing pipeline with:
        - Image preprocessing (deskew, denoise, contrast enhancement)
        - Region detection (tables, handwriting, printed text)
        - Layout-guided OCR routing
        """
        if self._document_pipeline is None:
            pipeline_config = {
                'enable_preprocessing': self.enable_preprocessing,
                'enable_region_detection': self.enable_region_detection,
                'use_vlm_classification': self.use_vlm_classification,
                'preprocessing': self.config.get('preprocessing', {}),
                'region_detection': self.config.get('region_detection', {}),
                'ocr': self.config.get('ocr', {}),
            }
            self._document_pipeline = DocumentPipeline(pipeline_config)
        return self._document_pipeline

    async def extract(
        self,
        pdf_path: Path,
        raw_text: str,
        template_id: Optional[str] = None,
        tables: Optional[List] = None,
        skip_preprocessing: bool = False
    ) -> ConsensusResult:
        """
        Run parallel extraction and merge results.

        Args:
            pdf_path: Path to PDF file
            raw_text: Pre-extracted text
            template_id: Optional template to use
            tables: Pre-extracted tables (if available)
            skip_preprocessing: Skip document preprocessing pipeline

        Returns:
            ConsensusResult with merged extractions
        """
        import time
        start_time = time.time()

        self.logger.info(f"Starting multi-method extraction for {pdf_path.name}")

        # ================================================================
        # PREPROCESSING PIPELINE (if enabled)
        # ================================================================
        pipeline_result: Optional[PipelineResult] = None
        enhanced_text = raw_text

        if self.enable_preprocessing and not skip_preprocessing:
            try:
                self.logger.info("Running document preprocessing pipeline...")
                pipeline_results = await self.document_pipeline.process_pdf(pdf_path)

                if pipeline_results:
                    pipeline_result = pipeline_results[0]  # Single document assumption

                    # Use enhanced text from pipeline if available
                    if pipeline_result.full_text:
                        enhanced_text = pipeline_result.full_text
                        self.logger.info(
                            f"Pipeline extracted {len(pipeline_result.full_text)} chars, "
                            f"{pipeline_result.total_regions} regions detected, "
                            f"avg confidence: {pipeline_result.average_confidence:.2f}"
                        )

                        # Log region breakdown
                        if pipeline_result.region_texts:
                            region_summary = ", ".join(
                                f"{k}: {len(v)} chars"
                                for k, v in pipeline_result.region_texts.items()
                            )
                            self.logger.debug(f"Region texts: {region_summary}")

                        # Log special content detection
                        if pipeline_result.has_handwriting:
                            self.logger.info("Handwriting detected - used TrOCR")
                        if pipeline_result.has_tables:
                            self.logger.info("Tables detected - used table-aware OCR")

            except Exception as e:
                self.logger.warning(f"Document pipeline failed, using raw text: {e}")
                enhanced_text = raw_text

        # ================================================================
        # PARALLEL EXTRACTION
        # ================================================================
        extraction_tasks = []

        # 1. Template extraction (if template available)
        if template_id:
            extraction_tasks.append(
                self._extract_with_template(pdf_path, enhanced_text, template_id, tables)
            )

        # 2. Table structure extraction
        extraction_tasks.append(
            self._extract_with_tables(pdf_path, tables)
        )

        # 3. Vector DB guided extraction (find similar documents)
        extraction_tasks.append(
            self._extract_with_vector_db(pdf_path, enhanced_text)
        )

        # 4. MedGemma LLM extraction (use enhanced text from pipeline)
        extraction_tasks.append(
            self._extract_with_llm(enhanced_text)
        )

        # 5. Pipeline-based OCR extraction (if pipeline ran successfully)
        # This uses region-specific OCR results from the preprocessing pipeline
        if pipeline_result and pipeline_result.ocr_result:
            extraction_tasks.append(
                self._extract_from_pipeline_ocr(pipeline_result)
            )
        # 6. PaddleOCR extraction (fallback if no pipeline, or for additional coverage)
        # Can be disabled via config: use_paddle_ocr=False
        elif self.config.get('use_paddle_ocr', True):
            extraction_tasks.append(
                self._extract_with_paddle_ocr(pdf_path)
            )

        # Run all extractions in parallel
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # Process results
        method_results: Dict[str, MethodResult] = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Extraction method failed: {result}")
                continue
            if result and isinstance(result, MethodResult):
                method_results[result.method_name] = result

        self.logger.info(
            f"Parallel extraction complete: {len(method_results)} methods succeeded"
        )

        # ================================================================
        # CONSENSUS MERGE
        # ================================================================
        consensus = self._merge_results(method_results)
        consensus.method_results = method_results
        consensus.extraction_time = time.time() - start_time
        consensus.calculate_overall_confidence()

        self.logger.info(
            f"Consensus merge complete: {consensus.fields_extracted} fields, "
            f"{consensus.fields_high_confidence} high confidence, "
            f"{consensus.fields_need_review} need review"
        )

        return consensus

    async def _extract_with_template(
        self,
        pdf_path: Path,
        raw_text: str,
        template_id: str,
        tables: Optional[List]
    ) -> MethodResult:
        """Extract using template matching."""
        import time
        start = time.time()

        try:
            from medical_ingestion.processors.lab.utils.template_loader import load_template
            from ..processors.lab.utils.parsing import (
                parse_lab_value,
                normalize_test_name,
                extract_reference_range
            )

            template = load_template(template_id)
            if not template:
                return MethodResult(
                    method_name="template",
                    values={},
                    units={},
                    confidence=0.0,
                    error=f"Template {template_id} not found"
                )

            values = {}
            units = {}

            # Extract tables if not provided
            if tables is None:
                tables = self.table_extractor.extract_tables(pdf_path)

            # Process each table
            for table in tables:
                for row in table.rows:
                    if len(row) < 2:
                        continue

                    # Try to match test name to template fields
                    test_name = row[0].strip() if row[0] else ""
                    normalized = normalize_test_name(test_name)

                    # Find matching template field
                    for field_def in template.get('fields', []):
                        aliases = field_def.get('aliases', [field_def.get('name', '')])
                        if normalized in [normalize_test_name(a) for a in aliases]:
                            # Parse value
                            raw_value = row[1].strip() if len(row) > 1 else ""
                            parsed = parse_lab_value(raw_value)

                            if parsed.get('value') is not None:
                                field_name = field_def.get('name', test_name)
                                values[field_name] = parsed['value']
                                units[field_name] = parsed.get('unit', field_def.get('unit', ''))
                            break

            confidence = 0.98 if values else 0.0

            return MethodResult(
                method_name="template",
                values=values,
                units=units,
                confidence=confidence,
                extraction_time=time.time() - start
            )

        except Exception as e:
            self.logger.error(f"Template extraction failed: {e}")
            return MethodResult(
                method_name="template",
                values={},
                units={},
                confidence=0.0,
                error=str(e),
                extraction_time=time.time() - start
            )

    async def _extract_with_tables(
        self,
        pdf_path: Path,
        tables: Optional[List]
    ) -> MethodResult:
        """Extract from table structure without template."""
        import time
        start = time.time()

        try:
            from ..processors.lab.utils.parsing import (
                parse_lab_value,
                normalize_test_name,
                is_likely_lab_test
            )

            if tables is None:
                tables = self.table_extractor.extract_tables(pdf_path)

            if not tables:
                return MethodResult(
                    method_name="table",
                    values={},
                    units={},
                    confidence=0.0,
                    error="No tables found"
                )

            values = {}
            units = {}

            for table in tables:
                # Detect which columns are test name, value, unit, reference
                col_types = self._detect_column_types(table.headers, table.rows[:3])

                name_col = col_types.get('name', 0)
                value_col = col_types.get('value', 1)
                unit_col = col_types.get('unit')

                for row in table.rows:
                    if len(row) <= max(name_col, value_col):
                        continue

                    test_name = row[name_col].strip() if row[name_col] else ""
                    if not test_name or not is_likely_lab_test(test_name):
                        continue

                    raw_value = row[value_col].strip() if row[value_col] else ""
                    parsed = parse_lab_value(raw_value)

                    if parsed.get('value') is not None:
                        normalized_name = normalize_test_name(test_name)
                        values[normalized_name] = parsed['value']

                        # Get unit
                        if unit_col is not None and unit_col < len(row):
                            units[normalized_name] = row[unit_col].strip()
                        else:
                            units[normalized_name] = parsed.get('unit', '')

            confidence = min(0.95, 0.80 + (len(values) * 0.01)) if values else 0.0

            return MethodResult(
                method_name="table",
                values=values,
                units=units,
                confidence=confidence,
                extraction_time=time.time() - start
            )

        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            return MethodResult(
                method_name="table",
                values={},
                units={},
                confidence=0.0,
                error=str(e),
                extraction_time=time.time() - start
            )

    async def _extract_with_vector_db(
        self,
        pdf_path: Path,
        raw_text: str
    ) -> MethodResult:
        """Extract using vector similarity to past extractions."""
        import time
        start = time.time()

        try:
            # Find similar documents
            similar_docs = await self.vector_store.find_similar(
                text=raw_text[:2000],  # Use first 2000 chars for embedding
                top_k=3
            )

            if not similar_docs or similar_docs[0]['similarity'] < 0.7:
                return MethodResult(
                    method_name="vector_db",
                    values={},
                    units={},
                    confidence=0.0,
                    error="No similar documents found"
                )

            # Use the most similar document's extraction as a guide
            best_match = similar_docs[0]
            values = {}
            units = {}

            # Transfer values from similar document's extraction
            for field_name, field_value in best_match.get('extracted_values', {}).items():
                # Look for the same field in current text
                found_value = self._find_value_in_text(
                    raw_text,
                    field_name,
                    field_value
                )
                if found_value is not None:
                    values[field_name] = found_value['value']
                    units[field_name] = found_value.get('unit', '')

            # Confidence based on similarity
            confidence = best_match['similarity'] * 0.95 if values else 0.0

            return MethodResult(
                method_name="vector_db",
                values=values,
                units=units,
                confidence=confidence,
                extraction_time=time.time() - start
            )

        except Exception as e:
            self.logger.debug(f"Vector DB extraction failed (may not be initialized): {e}")
            return MethodResult(
                method_name="vector_db",
                values={},
                units={},
                confidence=0.0,
                error=str(e),
                extraction_time=time.time() - start
            )

    async def _extract_with_llm(self, raw_text: str) -> MethodResult:
        """Extract using MedGemma LLM."""
        import time
        import json
        start = time.time()

        try:
            from ..processors.lab.utils.parsing import normalize_test_name

            prompt = """Extract all lab test results from this medical report.

Report text:
{text}

Return a JSON object with this structure:
{{
    "results": [
        {{"test_name": "...", "value": ..., "unit": "...", "reference_range": "..."}},
        ...
    ]
}}

Extract EVERY lab value you can find. For numeric values, return just the number.
JSON:""".format(text=raw_text[:3000])

            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.1,
                json_mode=True
            )

            # Parse response
            response_text = response.get('text', '')

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                from json_repair import repair_json
                data = repair_json(response_text, return_objects=True)

            values = {}
            units = {}

            for result in data.get('results', []):
                # Use consistent normalization (same as table extractor)
                test_name = result.get('test_name', '')
                normalized_name = normalize_test_name(test_name) if test_name else ''
                if normalized_name and result.get('value') is not None:
                    values[normalized_name] = result['value']
                    units[normalized_name] = result.get('unit', '')

            confidence = 0.87 if values else 0.0

            return MethodResult(
                method_name="llm",
                values=values,
                units=units,
                confidence=confidence,
                extraction_time=time.time() - start
            )

        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return MethodResult(
                method_name="llm",
                values={},
                units={},
                confidence=0.0,
                error=str(e),
                extraction_time=time.time() - start
            )

    async def _extract_with_paddle_ocr(self, pdf_path: Path) -> MethodResult:
        """
        Extract using PaddleOCR library.

        Uses the actual PaddleOCR library for high-quality OCR extraction,
        then parses the text to find lab values.
        """
        import time
        start = time.time()

        try:
            from ..extractors.paddle_ocr import convert_pdf_page_to_image
            from ..processors.lab.utils.parsing import (
                normalize_test_name,
                is_likely_lab_test
            )
            import re

            # Convert first page to image (most lab reports are single page)
            image_bytes = await convert_pdf_page_to_image(pdf_path, page_num=0)

            # Extract using PaddleOCR
            result = await self.paddle_ocr.extract_from_image_bytes_async(image_bytes)

            raw_text = result.get('text', '')
            lines = result.get('lines', [])

            values = {}
            units = {}

            # Pattern for lab values: test_name value [unit] [reference_range] [flag]
            value_pattern = re.compile(
                r'^(.+?)\s+'              # Test name (non-greedy)
                r'([\d.,]+)\s*'           # Value
                r'([a-zA-Z%/0-9E\^]+)?\s*'  # Unit (optional)
                r'(?:[\d.]+\s*-\s*[\d.]+)?\s*'  # Reference range (optional, don't capture)
                r'(H|L|HH|LL|High|Low)?$',     # Flag (optional)
                re.IGNORECASE
            )

            for line_info in lines:
                text = line_info.get('text', '').strip()
                if not text:
                    continue

                match = value_pattern.match(text)
                if match:
                    test_name = match.group(1).strip()

                    # Skip if not a likely lab test
                    if not is_likely_lab_test(test_name):
                        continue

                    value_str = match.group(2)
                    unit = match.group(3) or ''

                    try:
                        # Parse the value
                        value = float(value_str.replace(',', ''))

                        # Use consistent normalization
                        normalized_name = normalize_test_name(test_name)

                        values[normalized_name] = value
                        units[normalized_name] = unit

                    except ValueError:
                        pass

            # Also try parsing from raw text for any missed values
            if not values:
                # Fallback: look for patterns in raw text
                for line in raw_text.split('\n'):
                    line = line.strip()
                    match = value_pattern.match(line)
                    if match:
                        test_name = match.group(1).strip()
                        if is_likely_lab_test(test_name):
                            try:
                                value = float(match.group(2).replace(',', ''))
                                normalized_name = normalize_test_name(test_name)
                                if normalized_name not in values:
                                    values[normalized_name] = value
                                    units[normalized_name] = match.group(3) or ''
                            except ValueError:
                                pass

            # PaddleOCR is high accuracy for OCR
            confidence = 0.90 if values else 0.0

            self.logger.info(
                f"PaddleOCR extracted {len(values)} values in "
                f"{time.time() - start:.2f}s"
            )

            return MethodResult(
                method_name="paddle_ocr",
                values=values,
                units=units,
                confidence=confidence,
                extraction_time=time.time() - start
            )

        except ImportError as e:
            self.logger.warning(f"PaddleOCR not installed: {e}")
            return MethodResult(
                method_name="paddle_ocr",
                values={},
                units={},
                confidence=0.0,
                error="PaddleOCR not installed. Run: pip install paddlepaddle paddleocr",
                extraction_time=time.time() - start
            )
        except Exception as e:
            self.logger.error(f"PaddleOCR extraction failed: {e}")
            return MethodResult(
                method_name="paddle_ocr",
                values={},
                units={},
                confidence=0.0,
                error=str(e),
                extraction_time=time.time() - start
            )

    async def _extract_from_pipeline_ocr(
        self,
        pipeline_result: PipelineResult
    ) -> MethodResult:
        """
        Extract lab values from the document pipeline's OCR results.

        Uses region-aware OCR that has already been processed through:
        - Image preprocessing (deskew, denoise, contrast)
        - Region detection (tables, handwriting, printed text)
        - Layout-guided OCR routing (TrOCR for handwriting, PaddleOCR for print)

        This method extracts structured values from the enhanced text.
        """
        import time
        import re
        start = time.time()

        try:
            from ..processors.lab.utils.parsing import (
                normalize_test_name,
                is_likely_lab_test,
                parse_lab_value
            )

            values = {}
            units = {}

            # Get the full text and region-specific text
            full_text = pipeline_result.full_text or ""
            region_texts = pipeline_result.region_texts or {}

            # Pattern for lab values
            value_pattern = re.compile(
                r'^(.+?)\s+'
                r'([\d.,]+)\s*'
                r'([a-zA-Z%/0-9E\^]+)?\s*'
                r'(?:[\d.]+\s*-\s*[\d.]+)?\s*'
                r'(H|L|HH|LL|High|Low)?$',
                re.IGNORECASE
            )

            # Process table regions first (most structured)
            table_text = region_texts.get('table', '')
            if table_text:
                for line in table_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    match = value_pattern.match(line)
                    if match:
                        test_name = match.group(1).strip()
                        if is_likely_lab_test(test_name):
                            try:
                                value = float(match.group(2).replace(',', ''))
                                normalized_name = normalize_test_name(test_name)
                                values[normalized_name] = value
                                units[normalized_name] = match.group(3) or ''
                            except ValueError:
                                pass

            # Process printed text regions
            printed_text = region_texts.get('printed_text', '')
            if printed_text:
                for line in printed_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    match = value_pattern.match(line)
                    if match:
                        test_name = match.group(1).strip()
                        if is_likely_lab_test(test_name):
                            try:
                                value = float(match.group(2).replace(',', ''))
                                normalized_name = normalize_test_name(test_name)
                                if normalized_name not in values:
                                    values[normalized_name] = value
                                    units[normalized_name] = match.group(3) or ''
                            except ValueError:
                                pass

            # Fall back to full text parsing if no structured values found
            if not values:
                for line in full_text.split('\n'):
                    line = line.strip()
                    match = value_pattern.match(line)
                    if match:
                        test_name = match.group(1).strip()
                        if is_likely_lab_test(test_name):
                            try:
                                value = float(match.group(2).replace(',', ''))
                                normalized_name = normalize_test_name(test_name)
                                values[normalized_name] = value
                                units[normalized_name] = match.group(3) or ''
                            except ValueError:
                                pass

            # Calculate confidence based on pipeline quality
            base_confidence = pipeline_result.average_confidence
            if values:
                # Boost confidence based on region detection quality
                if pipeline_result.has_tables:
                    base_confidence = min(0.95, base_confidence + 0.05)
                confidence = max(0.80, base_confidence)
            else:
                confidence = 0.0

            self.logger.info(
                f"Pipeline OCR extracted {len(values)} values in "
                f"{time.time() - start:.2f}s (avg_confidence: {pipeline_result.average_confidence:.2f})"
            )

            return MethodResult(
                method_name="pipeline_ocr",
                values=values,
                units=units,
                confidence=confidence,
                extraction_time=time.time() - start
            )

        except Exception as e:
            self.logger.error(f"Pipeline OCR extraction failed: {e}")
            return MethodResult(
                method_name="pipeline_ocr",
                values={},
                units={},
                confidence=0.0,
                error=str(e),
                extraction_time=time.time() - start
            )

    def _merge_results(
        self,
        method_results: Dict[str, MethodResult]
    ) -> ConsensusResult:
        """
        Merge results from multiple methods using consensus logic.

        Agreement rules:
        - If all methods agree → unanimous, highest confidence
        - If 2+ agree → majority, high confidence
        - If only 1 found it → minority, medium confidence
        - If methods disagree → conflict, flag for review
        """
        consensus = ConsensusResult()

        # Collect all field names across all methods
        all_fields = set()
        for result in method_results.values():
            all_fields.update(result.values.keys())

        for field_name in all_fields:
            # Gather values from each method
            method_values = {}
            method_units = {}

            for method_name, result in method_results.items():
                if result.has_field(field_name):
                    method_values[method_name] = result.get_value(field_name)
                    method_units[method_name] = result.units.get(field_name, '')

            if not method_values:
                continue

            # Determine agreement level
            agreement, final_value, final_unit, contributing = self._calculate_agreement(
                method_values, method_units
            )

            # Calculate confidence based on agreement
            base_confidence = self.CONFIDENCE_LEVELS[agreement]

            # Boost confidence if high-accuracy methods agree
            if 'template' in contributing:
                base_confidence = min(0.99, base_confidence + 0.02)
            if 'paddle_ocr' in contributing:
                base_confidence = min(0.99, base_confidence + 0.02)
            if 'pipeline_ocr' in contributing:
                base_confidence = min(0.99, base_confidence + 0.03)  # Higher boost for preprocessed OCR

            # DYNAMIC CONFIDENCE: Validate value against expected ranges
            validation_boost = self._validate_value_plausibility(
                field_name, final_value, final_unit
            )
            if validation_boost > 0:
                # Boost minority confidence when value is plausible
                base_confidence = min(0.95, base_confidence + validation_boost)
                self.logger.debug(
                    f"Confidence boost for {field_name}: +{validation_boost:.2f} "
                    f"(value={final_value}, unit={final_unit})"
                )

            # Create consensus field
            cf = ConsensusField(
                field_name=field_name,
                final_value=final_value,
                final_unit=final_unit,
                confidence=base_confidence,
                agreement_level=agreement,
                contributing_methods=contributing,
                method_values=method_values
            )

            # Flag for review if low agreement
            if agreement in [AgreementLevel.CONFLICT, AgreementLevel.MINORITY]:
                cf.needs_review = True
                cf.review_reason = f"Low agreement: {agreement.value} ({contributing})"

            consensus.add_field(cf)

        return consensus

    def _calculate_agreement(
        self,
        method_values: Dict[str, Any],
        method_units: Dict[str, str]
    ) -> Tuple[AgreementLevel, Any, str, List[str]]:
        """
        Calculate agreement level and determine final value.

        Returns:
            (agreement_level, final_value, final_unit, contributing_methods)
        """
        if len(method_values) == 0:
            return AgreementLevel.NONE, None, '', []

        if len(method_values) == 1:
            method_name = list(method_values.keys())[0]
            return (
                AgreementLevel.MINORITY,
                method_values[method_name],
                method_units.get(method_name, ''),
                [method_name]
            )

        # Normalize values for comparison
        normalized_values = {}
        for method, value in method_values.items():
            normalized_values[method] = self._normalize_value(value)

        # Count occurrences of each value
        value_counts = Counter(normalized_values.values())
        most_common_value, most_common_count = value_counts.most_common(1)[0]

        # Find which methods contributed the most common value
        contributing = [
            m for m, v in normalized_values.items()
            if v == most_common_value
        ]

        # Get the best unit from contributing methods
        best_unit = ''
        # Priority: template > pipeline_ocr > paddle_ocr > table > llm
        for method in ['template', 'pipeline_ocr', 'paddle_ocr', 'table', 'llm']:
            if method in contributing and method in method_units:
                best_unit = method_units[method]
                break

        # Get original value from highest priority method
        final_value = None
        for method in ['template', 'pipeline_ocr', 'paddle_ocr', 'table', 'vector_db', 'llm']:
            if method in contributing:
                final_value = method_values[method]
                break

        # Determine agreement level
        total_methods = len(method_values)

        if most_common_count == total_methods:
            agreement = AgreementLevel.UNANIMOUS
        elif most_common_count >= 2:
            agreement = AgreementLevel.MAJORITY
        elif total_methods >= 2 and most_common_count == 1:
            agreement = AgreementLevel.CONFLICT
        else:
            agreement = AgreementLevel.MINORITY

        return agreement, final_value, best_unit, contributing

    def _validate_value_plausibility(
        self,
        field_name: str,
        value: Any,
        unit: str
    ) -> float:
        """
        Validate if a value is plausible for the given test.

        Returns a confidence boost (0.0 to 0.15) based on:
        - Value falls within expected physiological range
        - Unit matches expected unit for the test

        This helps boost MINORITY confidence when the value "makes sense"
        even if only one extraction method found it.
        """
        if value is None:
            return 0.0

        # Normalize field name for lookup
        field_lower = field_name.lower().replace('_', '').replace(' ', '')

        # Find matching expected range
        for test_pattern, (unit_patterns, min_val, max_val) in self.EXPECTED_RANGES.items():
            test_normalized = test_pattern.lower().replace('_', '')

            # Check if field name matches the test pattern
            if test_normalized in field_lower or field_lower in test_normalized:
                try:
                    numeric_value = float(value) if not isinstance(value, (int, float)) else value
                except (ValueError, TypeError):
                    return 0.0

                boost = 0.0

                # Check if value is in expected range
                if min_val <= numeric_value <= max_val:
                    boost += 0.10  # Value in range: +10%
                elif min_val * 0.5 <= numeric_value <= max_val * 1.5:
                    boost += 0.05  # Value slightly out of range but plausible: +5%
                else:
                    return 0.0  # Value way out of range, no boost

                # Check if unit matches expected patterns
                if unit:
                    unit_lower = unit.lower()
                    for unit_pattern in unit_patterns:
                        if unit_pattern.lower() in unit_lower:
                            boost += 0.05  # Unit matches: +5%
                            break

                return min(0.15, boost)  # Cap at 15% boost

        return 0.0  # Unknown test, no boost

    def _normalize_value(self, value: Any) -> Any:
        """Normalize value for comparison."""
        if value is None:
            return None

        # Convert to float if numeric string
        if isinstance(value, str):
            try:
                return float(value.replace(',', ''))
            except ValueError:
                return value.lower().strip()

        if isinstance(value, (int, float)):
            return round(float(value), 2)

        return value

    def _detect_column_types(
        self,
        headers: List[str],
        sample_rows: List[List[str]]
    ) -> Dict[str, int]:
        """Detect which columns contain test names, values, units, etc."""
        col_types = {'name': 0, 'value': 1}

        header_lower = [h.lower() for h in headers]

        # Check headers for clues
        for i, h in enumerate(header_lower):
            if any(kw in h for kw in ['test', 'name', 'component', 'analyte']):
                col_types['name'] = i
            elif any(kw in h for kw in ['result', 'value', 'level']):
                col_types['value'] = i
            elif any(kw in h for kw in ['unit', 'units']):
                col_types['unit'] = i
            elif any(kw in h for kw in ['reference', 'range', 'normal']):
                col_types['reference'] = i
            elif any(kw in h for kw in ['flag', 'status']):
                col_types['flag'] = i

        return col_types

    def _find_value_in_text(
        self,
        text: str,
        field_name: str,
        expected_value: Any
    ) -> Optional[Dict[str, Any]]:
        """Find a field value in text based on field name and expected value pattern."""
        import re
        from ..processors.lab.utils.parsing import parse_lab_value

        # Convert field name to search patterns
        search_name = field_name.replace('_', r'[\s_-]*')

        # Pattern: field_name followed by value
        pattern = rf'{search_name}\s*[:\s]\s*([\d.,]+)\s*(\w+)?'

        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value_str = match.group(1)
            unit = match.group(2) or ''
            parsed = parse_lab_value(value_str)
            if parsed.get('value') is not None:
                return {'value': parsed['value'], 'unit': unit}

        return None

    async def store_successful_extraction(
        self,
        pdf_path: Path,
        raw_text: str,
        extracted_values: Dict[str, Any],
        template_id: Optional[str] = None
    ):
        """Store successful extraction in vector DB for future reference."""
        try:
            await self.vector_store.store(
                text=raw_text[:2000],
                extracted_values=extracted_values,
                template_id=template_id,
                source_file=str(pdf_path)
            )
            self.logger.info(f"Stored extraction for future reference: {pdf_path.name}")
        except Exception as e:
            self.logger.debug(f"Failed to store extraction: {e}")
