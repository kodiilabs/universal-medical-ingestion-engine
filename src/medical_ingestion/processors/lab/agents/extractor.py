# ============================================================================
# FILE 1: src/medical_ingestion/processors/lab/agents/extractor.py
# ============================================================================
"""
Extraction Agent - Lab Values Extraction

Routes to appropriate extraction method based on template matching:
1. Template-based extraction (fast, high accuracy)
2. Table extraction with Camelot (structured PDFs)
3. MedGemma extraction (fallback for unknown formats)

Outputs structured lab values with provenance.
"""

from typing import Dict, Any, Optional
import json
import re

from json_repair import repair_json

from ....core.agent_base import Agent

# Confidence penalty when json_repair is used
JSON_REPAIR_CONFIDENCE_PENALTY = 0.10

# Minimum table extraction results before triggering MedGemma validation
# If table extraction yields fewer than this, results are "suspicious"
MIN_TABLE_RESULTS_THRESHOLD = 5

# Confidence boost when MedGemma validates table results
VALIDATION_CONFIDENCE_BOOST = 0.05


def calculate_extraction_confidence(
    value: Any,
    unit: str = None,
    reference_min: Any = None,
    reference_max: Any = None,
    extraction_method: str = "unknown",
    field_name: str = None
) -> float:
    """
    Calculate dynamic confidence for an extracted value.

    Factors considered:
    - Base confidence by extraction method
    - Bonus for having unit
    - Bonus for having reference range
    - Bonus for value falling within expected physiological range
    """
    # Base confidence by extraction method
    METHOD_BASE_CONFIDENCE = {
        "template": 0.95,
        "table": 0.88,
        "table_assisted": 0.82,
        "hybrid_medgemma": 0.78,
        "medgemma": 0.75,
        "vision": 0.72,
        "unknown": 0.70,
    }

    base = METHOD_BASE_CONFIDENCE.get(extraction_method, 0.70)

    # Bonus for having unit (+0.03)
    if unit and unit.strip():
        base += 0.03

    # Bonus for having reference range (+0.04)
    if reference_min is not None or reference_max is not None:
        base += 0.04

    # Check value plausibility if we have field name
    if field_name and value is not None:
        plausibility_boost = _check_value_plausibility(field_name, value, unit)
        base += plausibility_boost

    return min(0.98, base)  # Cap at 0.98


def _check_value_plausibility(field_name: str, value: Any, unit: str = None) -> float:
    """Check if value falls within expected physiological range."""
    EXPECTED_RANGES = {
        "wbc": (3.0, 15.0),
        "rbc": (3.5, 6.5),
        "hemoglobin": (10.0, 20.0),
        "hgb": (10.0, 20.0),
        "hematocrit": (30.0, 60.0),
        "hct": (30.0, 60.0),
        "mcv": (70.0, 110.0),
        "mch": (24.0, 36.0),
        "mchc": (30.0, 38.0),
        "platelets": (100.0, 500.0),
        "plt": (100.0, 500.0),
        "glucose": (50.0, 300.0),
        "bun": (5.0, 50.0),
        "creatinine": (0.3, 5.0),
        "sodium": (125.0, 155.0),
        "potassium": (2.5, 7.0),
        "chloride": (85.0, 115.0),
        "ast": (5.0, 150.0),
        "alt": (5.0, 150.0),
        "tsh": (0.1, 15.0),
    }

    try:
        numeric_value = float(value) if not isinstance(value, (int, float)) else value
    except (ValueError, TypeError):
        return 0.0

    field_lower = field_name.lower().replace('_', '').replace(' ', '')

    for test_key, (min_val, max_val) in EXPECTED_RANGES.items():
        if test_key in field_lower or field_lower in test_key:
            if min_val <= numeric_value <= max_val:
                return 0.05  # Value in expected range: +5%
            elif min_val * 0.3 <= numeric_value <= max_val * 2:
                return 0.02  # Plausible but borderline: +2%

    return 0.0  # Unknown test or out of range
from ....core.context import ProcessingContext, ExtractedValue
from ....extractors.table_extractor import TableExtractor
from ....extractors.text_extractor import TextExtractor
from ....medgemma.client import create_client
from ..utils.parsing import (
    parse_numeric_value,
    parse_value_and_flag,
    parse_reference_range,
    find_value_in_table,
    extract_abnormal_flag
)
from ..utils.template_loader import load_template


class ExtractionAgent(Agent):
    """
    Extracts lab values from PDF using best available method.
    
    Strategy (based on template matching result):
    - Template matched → Use template extraction rules
    - No template → Try table extraction (Camelot)
    - Table extraction fails → Use MedGemma
    
    Each extracted value includes:
    - Test name
    - Value
    - Unit
    - Reference range (if available)
    - Source location (provenance)
    - Confidence score
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.table_extractor = TableExtractor()
        self.text_extractor = TextExtractor()
        self.medgemma = create_client(config)

        # Consensus extraction mode (Unstract-inspired multi-method approach)
        self.use_consensus_extraction = config.get('use_consensus_extraction', False)
        self._extraction_engine = None

    @property
    def extraction_engine(self):
        """Lazy load ExtractionEngine for consensus mode."""
        if self._extraction_engine is None:
            from ....core.extraction_engine import ExtractionEngine
            self._extraction_engine = ExtractionEngine(self.config)
        return self._extraction_engine

    def get_name(self) -> str:
        return "ExtractionAgent"

    # Confidence threshold for "good enough" extraction (skip LLM)
    TEMPLATE_CONFIDENCE_THRESHOLD = 0.85
    MIN_EXPECTED_TESTS = 10  # Most lab panels have at least this many tests
    # If template extraction gets fewer than this ratio of expected fields, always assist
    MIN_EXTRACTION_RATIO = 0.6  # At least 60% of template fields should be extracted
    # Multi-panel keywords that indicate document has more tests than one template covers
    MULTI_PANEL_KEYWORDS = [
        'comp. metabolic', 'comprehensive metabolic', 'cmp',
        'lipid panel', 'lipid profile',
        'cbc with differential', 'cbc w/ diff',
        'thyroid panel', 'thyroid profile',
        'liver panel', 'hepatic panel',
        'iron', 'tibc',
        'testosterone', 'estradiol', 'dhea',
        'vitamin d', 'vitamin b12',
        'homocysteine', 'crp', 'c-reactive',
        'cortisol', 'ferritin',
        'uric acid', 'ggt', 'ldh',
    ]

    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract lab values using tiered approach:

        TIER 1: Template-first extraction (primary)
        - Use templates to extract based on vendor + test type
        - If confidence >= threshold → DONE, skip LLM entirely
        - Handles 70-85% of lab reports cleanly

        TIER 2: MedGemma as targeted helper (not full extraction)
        - Only used when template has issues:
          - Vendor not identified
          - Column layout breaks
          - Unusual naming
          - Missing tests
        - Ask bounded questions, not full JSON extraction
        - MedGemma helps understand, not decide

        CONSENSUS MODE (Unstract-inspired):
        - When use_consensus_extraction=True, runs multiple extraction methods in parallel
        - Template + Table + Vector DB + LLM all run simultaneously
        - Results merged using consensus logic (field-level agreement)
        - Higher accuracy on "near-miss" cases (5-10% improvement)

        OPTIMIZATION: Text and table extraction run in PARALLEL at the start
        to reduce latency on slower machines.

        Returns:
            {
                "decision": "extracted",
                "confidence": float,
                "reasoning": str,
                "method": "template" | "template_assisted" | "table" | "table_assisted" | "consensus",
                "values_extracted": int
            }
        """
        import asyncio

        # ====================================================================
        # CONSENSUS MODE: Multi-method extraction with agreement-based merge
        # ====================================================================
        if self.use_consensus_extraction:
            return await self._extract_with_consensus(context)

        # ====================================================================
        # EXTRACTION: Text in parallel, tables synchronously
        # ====================================================================
        # Note: Camelot uses Ghostscript subprocess calls which may have
        # thread-safety issues, so we run table extraction on main thread.

        # Extract text with page-level detail for multi-page awareness
        if not context.raw_text:
            loop = asyncio.get_event_loop()
            text_result = await loop.run_in_executor(
                None,
                self.text_extractor.extract_text_detailed,
                context.document_path
            )
            context.raw_text = text_result.text
            context.total_pages = text_result.page_count

            # Store page-level text for multi-page awareness
            for page in text_result.pages:
                context.add_page_text(page.page_number + 1, page.text)  # 1-indexed

            self.logger.info(
                f"Text extraction complete: {len(context.raw_text)} chars, "
                f"{text_result.page_count} pages"
            )

            if text_result.needs_ocr:
                context.add_warning(
                    f"Pages may need OCR: {text_result.pages_needing_ocr}"
                )

        # Extract tables synchronously (Camelot not thread-safe)
        self.logger.info("Extracting tables (synchronous for Camelot compatibility)")
        tables = self.table_extractor.extract_tables(context.document_path)
        context._extracted_tables = tables

        self.logger.info(
            f"Extraction complete: {len(context.raw_text)} chars text, "
            f"{len(tables) if tables else 0} tables, {context.total_pages} pages"
        )

        # ====================================================================
        # TIER 1: Template-first extraction
        # ====================================================================
        if context.template_id:
            self.logger.info(f"TIER 1: Template extraction with {context.template_id}")
            result = await self._extract_with_template(context)

            # Check if template extraction was successful enough
            if self._is_extraction_sufficient(result, context):
                self.logger.info(
                    f"Template extraction sufficient: {result['values_extracted']} values, "
                    f"confidence {result['confidence']:.2f} - skipping LLM"
                )
                return result

            # Template extraction incomplete - use MedGemma as targeted helper
            self.logger.info(
                f"Template extraction incomplete ({result['values_extracted']} values), "
                "using MedGemma as targeted helper"
            )
            result = await self._template_with_medgemma_assist(context, result)
            return result

        # ====================================================================
        # TIER 1b: Table extraction (no template matched)
        # ====================================================================
        self.logger.info("TIER 1b: Table extraction (no template)")
        result = await self._extract_with_tables(context)

        if self._is_extraction_sufficient(result, context):
            self.logger.info(
                f"Table extraction sufficient: {result['values_extracted']} values - skipping LLM"
            )
            return result

        # Table extraction incomplete - use MedGemma as targeted helper
        if result['values_extracted'] > 0:
            self.logger.info(
                f"Table extraction incomplete ({result['values_extracted']} values), "
                "using MedGemma as targeted helper"
            )
            result = await self._table_with_medgemma_assist(context, result)
        else:
            # No tables found at all - last resort full MedGemma
            self.logger.warning("No tables found, falling back to full MedGemma extraction")
            result = await self._extract_with_medgemma(context)

        return result

    async def _extract_with_consensus(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Multi-method extraction with consensus merge (Unstract-inspired).

        Runs template, table, vector DB, and LLM extraction in parallel,
        then merges results based on field-level agreement.

        Benefits:
        - Higher accuracy on "near-miss" cases (format variations)
        - Graceful degradation (95% vs 85% on similar layouts)
        - Field-level confidence based on agreement
        - Automatic flagging of low-agreement fields for review

        Returns:
            {
                "decision": "extracted",
                "confidence": float,
                "reasoning": str,
                "method": "consensus",
                "values_extracted": int,
                "agreement_stats": {...}
            }
        """
        import asyncio

        self.logger.info("CONSENSUS MODE: Running multi-method extraction")

        # Extract text if not available
        if not context.raw_text:
            loop = asyncio.get_event_loop()
            context.raw_text = await loop.run_in_executor(
                None,
                self.text_extractor.extract_text,
                context.document_path
            )

        # Extract tables (Camelot not thread-safe)
        tables = self.table_extractor.extract_tables(context.document_path)
        context._extracted_tables = tables

        # Run consensus extraction
        consensus_result = await self.extraction_engine.extract(
            pdf_path=context.document_path,
            raw_text=context.raw_text,
            template_id=context.template_id,
            tables=tables
        )

        # Convert consensus fields to ExtractedValues and add to context
        extracted_values = consensus_result.to_extracted_values()
        for ev in extracted_values:
            context.add_extracted_value(ev)

        # Track agreement statistics
        agreement_stats = {
            'unanimous': 0,
            'majority': 0,
            'minority': 0,
            'conflict': 0
        }
        for field in consensus_result.fields.values():
            agreement_stats[field.agreement_level.value] = (
                agreement_stats.get(field.agreement_level.value, 0) + 1
            )

        # Flag fields that need review
        fields_needing_review = [
            f.field_name for f in consensus_result.fields.values()
            if f.needs_review
        ]
        if fields_needing_review:
            context.requires_review = True
            context.review_reasons.append(
                f"Consensus extraction flagged {len(fields_needing_review)} fields: "
                f"{fields_needing_review[:5]}"
            )

        # Store successful extraction in vector DB for future reference
        if consensus_result.overall_confidence >= 0.85:
            await self._store_extraction_for_learning(context, consensus_result)

        self.logger.info(
            f"Consensus extraction complete: {consensus_result.fields_extracted} fields, "
            f"confidence {consensus_result.overall_confidence:.2f}, "
            f"agreement: {agreement_stats}"
        )

        return {
            "decision": "extracted",
            "confidence": consensus_result.overall_confidence,
            "reasoning": (
                f"Consensus extraction: {consensus_result.fields_extracted} fields from "
                f"{len(consensus_result.method_results)} methods. "
                f"High agreement: {agreement_stats['unanimous'] + agreement_stats['majority']}, "
                f"Low agreement: {agreement_stats['minority'] + agreement_stats['conflict']}"
            ),
            "method": "consensus",
            "values_extracted": consensus_result.fields_extracted,
            "agreement_stats": agreement_stats,
            "fields_need_review": len(fields_needing_review)
        }

    async def _store_extraction_for_learning(
        self,
        context: ProcessingContext,
        consensus_result
    ):
        """Store successful extraction in vector DB for future similar documents."""
        try:
            extracted_dict = {
                ev.field_name: ev.value
                for ev in context.extracted_values
            }
            await self.extraction_engine.store_successful_extraction(
                pdf_path=context.document_path,
                raw_text=context.raw_text[:2000],
                extracted_values=extracted_dict,
                template_id=context.template_id
            )
        except Exception as e:
            self.logger.debug(f"Failed to store extraction for learning: {e}")

    def _is_extraction_sufficient(self, result: Dict, context: ProcessingContext = None) -> bool:
        """
        Check if extraction is good enough to skip LLM assistance.

        Sufficient means:
        - At least MIN_EXPECTED_TESTS values extracted
        - Confidence above threshold
        - At least MIN_EXTRACTION_RATIO of template fields extracted (if using template)
        - Document is NOT a multi-panel report (always assist for multi-panel)
        """
        values_extracted = result.get('values_extracted', 0)
        confidence = result.get('confidence', 0)

        # Must have reasonable number of values
        if values_extracted < self.MIN_EXPECTED_TESTS:
            self.logger.debug(f"Insufficient: {values_extracted} < {self.MIN_EXPECTED_TESTS} tests")
            return False

        # Must have good confidence
        if confidence < self.TEMPLATE_CONFIDENCE_THRESHOLD:
            self.logger.debug(f"Insufficient: confidence {confidence:.2f} < {self.TEMPLATE_CONFIDENCE_THRESHOLD}")
            return False

        # Check extraction ratio vs template fields
        template_fields = result.get('template_fields', 0)
        if template_fields > 0:
            ratio = values_extracted / template_fields
            if ratio < self.MIN_EXTRACTION_RATIO:
                self.logger.debug(
                    f"Insufficient: extraction ratio {ratio:.2f} < {self.MIN_EXTRACTION_RATIO} "
                    f"({values_extracted}/{template_fields} fields)"
                )
                return False

        # Check for multi-panel document - always assist for comprehensive reports
        if context and context.raw_text:
            text_lower = context.raw_text.lower()
            panel_count = sum(1 for kw in self.MULTI_PANEL_KEYWORDS if kw in text_lower)
            if panel_count >= 3:
                self.logger.info(
                    f"Multi-panel document detected ({panel_count} panel indicators) - "
                    "will use MedGemma assist to find additional tests"
                )
                return False

        return True

    async def _template_with_medgemma_assist(
        self,
        context: ProcessingContext,
        template_result: Dict
    ) -> Dict[str, Any]:
        """
        Use MedGemma as a targeted helper for template extraction issues.

        Strategy:
        1. Ask MedGemma bounded questions (not full JSON extraction)
        2. Use answers to fill gaps in template extraction
        3. MedGemma helps understand, not decide
        """
        text = context.raw_text or self.text_extractor.extract_text(context.document_path)
        extracted_names = [v.field_name for v in context.extracted_values]

        # BOUNDED QUERY 1: What tests are in this document?
        all_tests = await self._medgemma_list_tests(text)

        if not all_tests:
            self.logger.warning("MedGemma couldn't identify tests - using template results only")
            return template_result

        # Find tests that template missed
        missing_tests = []
        for test in all_tests:
            test_normalized = test.lower().replace(' ', '_')
            if not any(test_normalized in name or name in test_normalized for name in extracted_names):
                missing_tests.append(test)

        if not missing_tests:
            self.logger.info("Template found all tests - no MedGemma assistance needed")
            return template_result

        self.logger.info(f"Template missed {len(missing_tests)} tests: {missing_tests[:5]}...")

        # BOUNDED QUERY 2: Get values for missing tests only
        missing_values = await self._medgemma_get_specific_values(text, missing_tests)

        # Add missing values to context
        added_count = 0
        for test_name, value_data in missing_values.items():
            if value_data.get('value') is not None:
                extracted = ExtractedValue(
                    field_name=test_name.lower().replace(' ', '_'),
                    value=value_data.get('value'),
                    unit=value_data.get('unit', ''),
                    confidence=0.80,  # Slightly lower than template
                    extraction_method="template_assisted",
                    reference_min=value_data.get('reference_min'),
                    reference_max=value_data.get('reference_max'),
                    abnormal_flag=value_data.get('abnormal_flag')
                )
                context.add_extracted_value(extracted)
                added_count += 1

        total = template_result['values_extracted'] + added_count

        self.logger.info(
            f"Template-assisted extraction complete: {template_result['values_extracted']} from template, "
            f"{added_count} from MedGemma assist = {total} total"
        )

        return {
            "decision": "extracted",
            "confidence": 0.90 if total > 0 else 0.0,
            "reasoning": f"Template: {template_result['values_extracted']}, MedGemma assist: {added_count}",
            "method": "template_assisted",
            "values_extracted": total,
            "template_count": template_result['values_extracted'],
            "assist_count": added_count
        }

    async def _table_with_medgemma_assist(
        self,
        context: ProcessingContext,
        table_result: Dict
    ) -> Dict[str, Any]:
        """
        Use MedGemma as a targeted helper for table extraction issues.
        """
        text = context.raw_text or self.text_extractor.extract_text(context.document_path)
        extracted_names = [v.field_name for v in context.extracted_values]

        # BOUNDED QUERY: What tests are in this document that we might have missed?
        all_tests = await self._medgemma_list_tests(text)

        if not all_tests:
            return table_result

        # Find missing tests
        missing_tests = []
        for test in all_tests:
            test_normalized = test.lower().replace(' ', '_')
            if not any(test_normalized in name or name in test_normalized for name in extracted_names):
                missing_tests.append(test)

        if not missing_tests:
            return table_result

        self.logger.info(f"Table missed {len(missing_tests)} tests, getting values via MedGemma")

        # Get values for missing tests
        missing_values = await self._medgemma_get_specific_values(text, missing_tests)

        added_count = 0
        for test_name, value_data in missing_values.items():
            if value_data.get('value') is not None:
                field_name = test_name.lower().replace(' ', '_')
                extracted = ExtractedValue(
                    field_name=field_name,
                    value=value_data.get('value'),
                    unit=value_data.get('unit', ''),
                    confidence=calculate_extraction_confidence(
                        value=value_data.get('value'),
                        unit=value_data.get('unit', ''),
                        reference_min=value_data.get('reference_min'),
                        reference_max=value_data.get('reference_max'),
                        extraction_method="table_assisted",
                        field_name=field_name
                    ),
                    extraction_method="table_assisted",
                    reference_min=value_data.get('reference_min'),
                    reference_max=value_data.get('reference_max'),
                    abnormal_flag=value_data.get('abnormal_flag')
                )
                context.add_extracted_value(extracted)
                added_count += 1

        total = table_result['values_extracted'] + added_count

        return {
            "decision": "extracted",
            "confidence": 0.85 if total > 0 else 0.0,
            "reasoning": f"Table: {table_result['values_extracted']}, MedGemma assist: {added_count}",
            "method": "table_assisted",
            "values_extracted": total
        }

    # ========================================================================
    # BOUNDED MEDGEMMA QUERIES - Ask specific questions, not full extraction
    # ========================================================================

    async def _medgemma_list_tests(self, text: str) -> list:
        """
        BOUNDED QUERY: List all test names in the document.

        Returns simple list of test names - MedGemma helps identify, not extract.
        """
        text_sample = text[:8000]

        prompt = f"""List ALL lab test names found in this document. Return ONLY a JSON array of test names.

Document:
{text_sample}

Return format: ["WBC", "RBC", "Hemoglobin", "Hematocrit", "MCV", ...]

Include every test you can identify. Just the names, no values.
JSON array:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.1,
                json_mode=True
            )

            result_text = response.get('text', '').strip()

            if not result_text:
                self.logger.warning("MedGemma returned empty response for list_tests")
                return []

            # Try 1: Direct JSON parse
            try:
                tests = json.loads(result_text)
                if isinstance(tests, list):
                    self.logger.debug(f"MedGemma identified {len(tests)} tests")
                    return tests
                # Handle case where model returns {"tests": [...]} or nested structure
                if isinstance(tests, dict):
                    for key in ['tests', 'test_names', 'names', 'results']:
                        if key in tests and isinstance(tests[key], list):
                            return tests[key]
                    # Handle case where model returns {"WBC": "6.4 x10E3/uL", ...}
                    # Extract keys as test names if values look like lab results
                    if tests and all(isinstance(v, (str, int, float, dict)) for v in tests.values()):
                        test_names = list(tests.keys())
                        self.logger.debug(f"Extracted {len(test_names)} test names from dict keys")
                        return test_names
            except json.JSONDecodeError as e:
                self.logger.debug(f"Direct JSON parse failed: {e}")

            # Try 2: json_repair
            try:
                repaired = repair_json(result_text, return_objects=True)
                if isinstance(repaired, list):
                    self.logger.debug(f"json_repair fixed list_tests response: {len(repaired)} tests")
                    return repaired
                if isinstance(repaired, dict):
                    for key in ['tests', 'test_names', 'names', 'results']:
                        if key in repaired and isinstance(repaired[key], list):
                            return repaired[key]
                    # Handle dict with test names as keys
                    if repaired and all(isinstance(v, (str, int, float, dict)) for v in repaired.values()):
                        test_names = list(repaired.keys())
                        self.logger.debug(f"json_repair: extracted {len(test_names)} test names from dict keys")
                        return test_names
            except Exception as e:
                self.logger.debug(f"json_repair failed: {e}")

            # Try 3: Extract array with regex
            match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if match:
                try:
                    tests = json.loads(match.group())
                    if isinstance(tests, list):
                        return tests
                except json.JSONDecodeError:
                    # Try repair on extracted array
                    try:
                        repaired = repair_json(match.group(), return_objects=True)
                        if isinstance(repaired, list):
                            return repaired
                    except Exception:
                        pass

            self.logger.warning(f"Could not parse list_tests response: {result_text[:200]}...")
            return []

        except Exception as e:
            self.logger.error(f"MedGemma list_tests failed: {e}")
            return []

    async def _medgemma_get_specific_values(self, text: str, test_names: list) -> Dict:
        """
        BOUNDED QUERY: Get values for specific tests only.

        Instead of asking for ALL values, ask for SPECIFIC missing ones.
        """
        if not test_names:
            return {}

        # Limit to reasonable number
        test_names = test_names[:20]
        text_sample = text[:8000]

        prompt = f"""Find the values for these SPECIFIC tests in the document:
{json.dumps(test_names)}

Document:
{text_sample}

Return ONLY a JSON object with the test values you find:
{{
  "WBC": {{"value": 5.4, "unit": "x10E3/uL", "reference_min": 3.4, "reference_max": 10.8, "abnormal_flag": null}},
  "RBC": {{"value": 4.96, "unit": "x10E6/uL", "reference_min": 3.77, "reference_max": 5.28, "abnormal_flag": null}}
}}

Only include tests you can find in the document. Use null for missing fields.
JSON:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1,
                json_mode=True
            )

            result_text = response.get('text', '').strip()

            if not result_text:
                self.logger.warning("MedGemma returned empty response for get_specific_values")
                return {}

            # Try 1: Direct JSON parse
            try:
                values = json.loads(result_text)
                if isinstance(values, dict):
                    self.logger.debug(f"MedGemma found values for {len(values)} tests")
                    return values
            except json.JSONDecodeError as e:
                self.logger.debug(f"Direct JSON parse failed: {e}")

            # Try 2: json_repair
            try:
                repaired = repair_json(result_text, return_objects=True)
                if isinstance(repaired, dict):
                    self.logger.debug(f"json_repair fixed get_specific_values: {len(repaired)} tests")
                    return repaired
            except Exception as e:
                self.logger.debug(f"json_repair failed: {e}")

            # Try 3: Extract JSON object with regex (find outermost braces)
            try:
                start = result_text.find('{')
                if start != -1:
                    depth = 0
                    end = start
                    for i, char in enumerate(result_text[start:], start=start):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end = i
                                break

                    json_str = result_text[start:end + 1]
                    try:
                        values = json.loads(json_str)
                        if isinstance(values, dict):
                            return values
                    except json.JSONDecodeError:
                        # Try repair on extracted object
                        repaired = repair_json(json_str, return_objects=True)
                        if isinstance(repaired, dict):
                            return repaired
            except Exception:
                pass

            self.logger.warning(f"Could not parse get_specific_values response: {result_text[:200]}...")
            return {}

        except Exception as e:
            self.logger.error(f"MedGemma get_specific_values failed: {e}")
            return {}

    async def _parse_tables_with_medgemma(self, tables: list) -> list:
        """
        Send Camelot-extracted tables to MedGemma for intelligent parsing.

        This handles the common issue where Camelot extracts table structure
        but columns are misaligned, merged, or hard to parse programmatically.

        MedGemma can understand the table semantically and extract correct values
        even when column alignment is broken.

        Returns list of dicts: [{test_name, value, unit, reference_min, reference_max, abnormal_flag}]
        """
        if not tables:
            return []

        # Format tables as readable text for MedGemma
        table_text = []
        for i, table in enumerate(tables):
            table_text.append(f"=== TABLE {i+1} (Page {table.page_number}) ===")
            if table.headers:
                table_text.append("HEADERS: " + " | ".join(table.headers))
            table_text.append("ROWS:")
            for row in table.rows[:50]:  # Limit rows
                table_text.append(" | ".join(str(cell) for cell in row))
            table_text.append("")

        tables_str = "\n".join(table_text)

        # Limit total text
        if len(tables_str) > 10000:
            tables_str = tables_str[:10000] + "\n... (truncated)"

        prompt = f"""You are a medical lab report parser. I have extracted table data from a PDF but the columns may be misaligned or merged.

Your job is to CORRECTLY identify each lab test and its value, even if the table structure is messy.

TABLE DATA:
{tables_str}

COMMON COLUMN PATTERNS IN LAB REPORTS:
- Test Name | Result | Flag | Units | Reference Range | Lab
- Test Name | Method | Result | Units | Reference Range (Sterling Accuris format)
- Sometimes columns are merged: "WBC 6.4" or "6.4 x10E3/uL"
- Flags are H (High), L (Low), or empty
- Reference ranges appear in various formats - ALWAYS look for them

EXTRACT ALL LAB TESTS. Return ONLY this JSON format:
{{
  "results": [
    {{"test_name": "WBC", "value": 6.4, "unit": "x10E3/uL", "reference_min": 3.4, "reference_max": 10.8, "abnormal_flag": null}},
    {{"test_name": "RBC", "value": 4.33, "unit": "x10E6/uL", "reference_min": 3.77, "reference_max": 5.28, "abnormal_flag": null}},
    {{"test_name": "Hemoglobin", "value": 12.7, "unit": "g/dL", "reference_min": 11.1, "reference_max": 15.9, "abnormal_flag": null}}
  ]
}}

CRITICAL RULES:
1. The VALUE is the NUMERIC RESULT, not the unit (e.g., 6.4, NOT 103 from "x10E3")
2. Units like "x10E3/uL" mean "times 10^3 per microliter" - they are NOT the value
3. ALWAYS extract reference ranges - look for them after the unit:
   - "3.4 - 10.8" or "3.4-10.8" → reference_min: 3.4, reference_max: 10.8
   - "13.0 - 16.5" → reference_min: 13.0, reference_max: 16.5
   - "<200" → reference_min: null, reference_max: 200
   - ">60" → reference_min: 60, reference_max: null
4. If a column looks like "6.4 x10E3/uL", the value is 6.4 and unit is x10E3/uL
5. Extract EVERY test you can find
6. Skip blood group/type entries (ABO, Rh) - only numeric lab tests

JSON:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.1,
                json_mode=True
            )

            result_text = response.get('text', '').strip()

            if not result_text:
                self.logger.warning("MedGemma table parsing returned empty response")
                return []

            # Parse JSON with fallbacks
            lab_results = self._parse_lab_json(result_text)

            if lab_results:
                self.logger.info(f"MedGemma parsed {len(lab_results)} results from tables")

                # Validate results - check for obvious parsing errors
                valid_results = []
                for r in lab_results:
                    value = r.get('value')
                    if value is not None:
                        try:
                            # Check for suspicious values (103, 106 from unit parsing)
                            v = float(value)
                            unit = r.get('unit', '')
                            # If value matches digits in unit's scientific notation, it's wrong
                            if unit and 'E' in unit.upper():
                                import re
                                unit_digits = re.sub(r'[^\d]', '', unit)
                                if str(int(v)) == unit_digits:
                                    self.logger.warning(
                                        f"Rejecting {r.get('test_name')}: value {v} matches unit digits {unit_digits}"
                                    )
                                    continue
                            valid_results.append(r)
                        except (ValueError, TypeError):
                            valid_results.append(r)  # Keep non-numeric values

                return valid_results

            return []

        except Exception as e:
            self.logger.error(f"MedGemma table parsing failed: {e}")
            return []

    def _is_extraction_suspicious(self, context: ProcessingContext, result: Dict) -> bool:
        """
        Detect if table extraction results are suspicious and need MedGemma validation.

        Suspicious indicators:
        - Too few values extracted (< MIN_TABLE_RESULTS_THRESHOLD)
        - No units extracted (lab reports always have units)
        - All values have same confidence (no variation = possible parsing error)
        """
        extracted_count = result.get('values_extracted', 0)

        # Too few results for a typical lab report
        if extracted_count < MIN_TABLE_RESULTS_THRESHOLD:
            self.logger.debug(f"Suspicious: only {extracted_count} values (threshold: {MIN_TABLE_RESULTS_THRESHOLD})")
            return True

        # Check if any units were extracted
        values_with_units = sum(1 for v in context.extracted_values if v.unit and v.unit.strip())
        if values_with_units == 0 and extracted_count > 0:
            self.logger.debug("Suspicious: no units extracted")
            return True

        # Check if values look reasonable (not all the same - which indicates parsing error)
        unique_values = set(v.value for v in context.extracted_values if v.value is not None)
        if extracted_count > 5 and len(unique_values) < 3:
            self.logger.debug(f"Suspicious: only {len(unique_values)} unique values from {extracted_count} extractions")
            return True

        return False

    async def _hybrid_extraction(self, context: ProcessingContext, table_result: Dict) -> Dict[str, Any]:
        """
        Hybrid extraction: combine table extraction with MedGemma validation.

        Strategy:
        1. Keep table-extracted values
        2. Run MedGemma extraction on raw text
        3. Merge results: MedGemma fills gaps, validates existing values
        4. Flag conflicts for review
        """
        self.logger.info("Running hybrid extraction (table + MedGemma)")

        # Store table-extracted values (we'll compare later)
        table_values = {v.field_name: v for v in context.extracted_values}

        # Clear extracted values - we'll rebuild with merged results
        context.extracted_values = []

        # Get MedGemma extraction with page-aware text
        text = context.get_full_text_with_page_markers() if context.page_text else context.raw_text
        if not text:
            text = self.text_extractor.extract_text(context.document_path)
        medgemma_result = await self._run_medgemma_extraction(text, context.total_pages)

        if not medgemma_result:
            # MedGemma failed - restore table values
            context.extracted_values = list(table_values.values())
            self.logger.warning("MedGemma extraction failed, using table values only")
            return table_result

        # Merge results
        merged_count = 0
        validated_count = 0
        added_count = 0
        conflicts = []

        for mg_result in medgemma_result:
            field_name = mg_result.get('test_name', '').lower().replace(' ', '_')
            if not field_name:
                continue

            # Check if we have a table-extracted value for this test
            if field_name in table_values:
                table_val = table_values[field_name]
                mg_value = mg_result.get('value')

                # Compare values
                if table_val.value is not None and mg_value is not None:
                    try:
                        table_num = float(table_val.value)
                        mg_num = float(mg_value)

                        # Check if values match (within 1% tolerance)
                        if abs(table_num - mg_num) / max(abs(mg_num), 0.001) < 0.01:
                            # Values match - boost confidence
                            validated_value = ExtractedValue(
                                field_name=field_name,
                                value=table_val.value,
                                unit=table_val.unit or mg_result.get('unit', ''),
                                confidence=min(0.98, table_val.confidence + VALIDATION_CONFIDENCE_BOOST),
                                extraction_method="hybrid_validated",
                                source_page=table_val.source_page,
                                source_location=table_val.source_location,
                                source_text=table_val.source_text,
                                reference_min=table_val.reference_min or mg_result.get('reference_min'),
                                reference_max=table_val.reference_max or mg_result.get('reference_max'),
                                abnormal_flag=table_val.abnormal_flag or mg_result.get('abnormal_flag')
                            )
                            context.add_extracted_value(validated_value)
                            validated_count += 1
                        else:
                            # Values conflict - use MedGemma (likely more accurate for text parsing)
                            # but flag for review
                            conflicts.append({
                                'test': field_name,
                                'table_value': table_num,
                                'medgemma_value': mg_num
                            })

                            # Use MedGemma value but lower confidence
                            conflict_value = ExtractedValue(
                                field_name=field_name,
                                value=mg_value,
                                unit=mg_result.get('unit', ''),
                                confidence=0.70,  # Lower due to conflict
                                extraction_method="hybrid_conflict_medgemma",
                                reference_min=mg_result.get('reference_min'),
                                reference_max=mg_result.get('reference_max'),
                                abnormal_flag=mg_result.get('abnormal_flag'),
                                validation_conflict=True
                            )
                            context.add_extracted_value(conflict_value)
                            merged_count += 1
                    except (ValueError, TypeError):
                        # Non-numeric comparison - use MedGemma
                        pass

                # Remove from table_values as processed
                del table_values[field_name]
            else:
                # New value from MedGemma - add it
                new_value = ExtractedValue(
                    field_name=field_name,
                    value=mg_result.get('value'),
                    unit=mg_result.get('unit', ''),
                    confidence=calculate_extraction_confidence(
                        value=mg_result.get('value'),
                        unit=mg_result.get('unit', ''),
                        reference_min=mg_result.get('reference_min'),
                        reference_max=mg_result.get('reference_max'),
                        extraction_method="hybrid_medgemma",
                        field_name=field_name
                    ),
                    extraction_method="hybrid_medgemma",
                    reference_min=mg_result.get('reference_min'),
                    reference_max=mg_result.get('reference_max'),
                    abnormal_flag=mg_result.get('abnormal_flag')
                )
                context.add_extracted_value(new_value)
                added_count += 1

        # Add any remaining table-only values
        for field_name, table_val in table_values.items():
            context.add_extracted_value(table_val)
            merged_count += 1

        total_extracted = len(context.extracted_values)

        # Log conflicts and flag for review
        if conflicts:
            self.logger.warning(f"Found {len(conflicts)} value conflicts between table and MedGemma")
            context.add_warning(f"{len(conflicts)} extraction conflicts detected - review recommended")
            context.requires_review = True
            context.review_reasons.append(
                f"Hybrid extraction found {len(conflicts)} conflicting values"
            )

        # Calculate overall confidence
        if total_extracted > 0:
            avg_confidence = sum(v.confidence for v in context.extracted_values) / total_extracted
        else:
            avg_confidence = 0.0

        self.logger.info(
            f"Hybrid extraction complete: {total_extracted} values "
            f"(validated: {validated_count}, added: {added_count}, conflicts: {len(conflicts)})"
        )

        return {
            "decision": "extracted",
            "confidence": avg_confidence,
            "reasoning": (
                f"Hybrid extraction: {validated_count} validated, "
                f"{added_count} added by MedGemma, {len(conflicts)} conflicts"
            ),
            "method": "hybrid",
            "values_extracted": total_extracted,
            "validated_count": validated_count,
            "added_count": added_count,
            "conflict_count": len(conflicts),
            "conflicts": conflicts
        }

    async def _run_medgemma_extraction(self, text: str, total_pages: int = 1) -> list:
        """
        Run MedGemma extraction and return parsed results.

        Multi-page aware: If text contains page markers (=== PAGE X of Y ===),
        instructs MedGemma to extract from ALL pages.

        Returns list of dicts: [{test_name, value, unit, reference_min, reference_max, abnormal_flag, page}]
        """
        # Check if text is too short (likely scanned PDF)
        MIN_TEXT_FOR_EXTRACTION = 500
        if len(text.strip()) < MIN_TEXT_FOR_EXTRACTION:
            self.logger.warning(
                f"Text too short ({len(text)} chars) - PDF may be scanned/image-based. "
                "MedGemma text extraction may fail."
            )

        # Use more text for multi-page documents
        # Typical lab report: ~2000-4000 chars per page
        max_chars = min(20000, 6000 * total_pages)  # Scale with pages
        text_sample = text[:max_chars]

        # Detect if text has page markers
        has_page_markers = "=== PAGE" in text_sample
        page_info = f"This is a {total_pages}-page document." if total_pages > 1 else ""

        self.logger.info(
            f"MedGemma extraction using {len(text_sample)} chars of {len(text)} total "
            f"({total_pages} pages)"
        )

        prompt = f"""You are a medical lab report parser. Your job is to extract ALL lab test results.

{page_info}
{'IMPORTANT: The document has PAGE MARKERS (=== PAGE X ===). Extract tests from ALL pages.' if has_page_markers else ''}

DOCUMENT TEXT:
{text_sample}

TASK: Extract EVERY lab test result from ALL PAGES. Lab reports typically have 20-60+ tests across multiple panels:
- CBC (Complete Blood Count): ~15 tests (WBC, RBC, Hemoglobin, Hematocrit, MCV, MCH, MCHC, RDW, Platelets, etc.)
- CMP (Comprehensive Metabolic Panel): ~14 tests (Glucose, BUN, Creatinine, Sodium, Potassium, etc.)
- Lipid Panel: ~5 tests (Total Cholesterol, HDL, LDL, Triglycerides, etc.)
- Thyroid, Liver, etc.

OUTPUT FORMAT - Return ONLY this JSON:
{{
  "results": [
    {{"test_name": "WBC", "value": 6.4, "unit": "x10E3/uL", "reference_min": 3.4, "reference_max": 10.8, "abnormal_flag": null, "page": 1}},
    {{"test_name": "RBC", "value": 4.33, "unit": "x10E6/uL", "reference_min": 3.77, "reference_max": 5.28, "abnormal_flag": null, "page": 1}},
    {{"test_name": "Hemoglobin", "value": 12.7, "unit": "g/dL", "reference_min": 11.1, "reference_max": 15.9, "abnormal_flag": null, "page": 1}},
    {{"test_name": "Hematocrit", "value": 38.5, "unit": "%", "reference_min": 34.0, "reference_max": 46.6, "abnormal_flag": null, "page": 1}},
    {{"test_name": "MCV", "value": 88.9, "unit": "fL", "reference_min": 79, "reference_max": 97, "abnormal_flag": null, "page": 2}}
  ]
}}

CRITICAL RULES:
1. Extract EVERY test from ALL PAGES - do not stop early or skip pages
2. Use EXACT values from document - never round or estimate
3. Copy units exactly (g/dL, x10E3/uL, mg/dL, mmol/L, %, fL, pg, etc.)
4. ALWAYS extract reference ranges - they appear AFTER the unit in various formats:
   - "13.0 - 16.5" or "13.0-16.5" → reference_min: 13.0, reference_max: 16.5
   - "4000 - 10000" → reference_min: 4000, reference_max: 10000
   - "<200" or "< 200" → reference_min: null, reference_max: 200
   - ">60" or "> 60" → reference_min: 60, reference_max: null
   - Column formats: "Test Method Value Unit Reference" - look for numbers after the unit
5. Flags: "H" for high, "L" for low, "HH"/"LL" for critical, null if normal
6. Include descriptive results as strings: "Normochromic Normocytic" → value: "Normochromic Normocytic"
7. Track which page each test came from using the "page" field
8. DO NOT include blood group/type (ABO, Rh) - only extract numeric lab test results

JSON:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=6000,  # Increased for 50+ results
                temperature=0.1,
                json_mode=True
            )

            result_text = response.get('text', '')
            self.logger.debug(f"MedGemma response: {len(result_text)} chars")

            # Parse JSON with multiple fallbacks
            lab_results = self._parse_lab_json(result_text)

            if lab_results:
                self.logger.info(f"MedGemma extracted {len(lab_results)} lab results")
            else:
                self.logger.warning("MedGemma extraction returned no parseable results")

            return lab_results

        except Exception as e:
            self.logger.error(f"MedGemma extraction failed: {e}")
            return None

    def _parse_lab_json(self, text: str) -> list:
        """Parse lab results JSON with multiple fallback strategies."""
        # Try 1: Direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ['results', 'lab_results', 'data', 'tests']:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
        except json.JSONDecodeError:
            pass

        # Try 2: json_repair
        try:
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, list):
                self.logger.warning("Used json_repair on MedGemma output")
                return repaired
            if isinstance(repaired, dict):
                for key in ['results', 'lab_results', 'data', 'tests']:
                    if key in repaired and isinstance(repaired[key], list):
                        self.logger.warning("Used json_repair on MedGemma output")
                        return repaired[key]
        except Exception:
            pass

        # Try 3: Extract JSON block with regex
        try:
            # Find array
            match = re.search(r'\[\s*\{[^[\]]*\}\s*(?:,\s*\{[^[\]]*\}\s*)*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        self.logger.warning(f"All JSON parse attempts failed. Response start: {text[:300]}...")
        return None
    
    async def _extract_with_template(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract using template rules (fastest, most accurate).

        Template contains field mappings and extraction rules.
        Handles various lab report formats including LabCorp Enterprise Report.
        """
        self.logger.info(f"Extracting with template: {context.template_id}")

        # Load template
        template = load_template(context.template_id)

        # Use pre-extracted tables from parallel extraction (or extract if not available)
        tables = getattr(context, '_extracted_tables', None)
        if tables is None:
            tables = self.table_extractor.extract_tables(context.document_path)

        if not tables:
            self.logger.warning("No tables found in PDF")
            return {
                "decision": "extraction_failed",
                "confidence": 0.0,
                "reasoning": "No tables found in PDF",
                "method": "template",
                "values_extracted": 0
            }

        # Check if template indicates flags are embedded in result column
        layout_sig = template.get('layout_signature', {})
        flag_in_result = layout_sig.get('flag_in_result_column', False)

        # Extract values using template field mappings
        # Process ALL tables to find all values
        extracted_count = 0
        found_fields = set()

        for table in tables:
            for field_name, field_config in template.get('field_mappings', {}).items():
                # Skip if already found
                if field_name in found_fields:
                    continue

                # Try to find by pdf_name first, then by aliases
                search_names = [field_config.get('pdf_name', field_name)]
                search_names.extend(field_config.get('aliases', []))

                value_data = None
                for search_name in search_names:
                    value_data = find_value_in_table(
                        table,
                        search_name,
                        value_column_index=field_config.get('value_column', 1)
                    )
                    if value_data:
                        break

                if value_data:
                    value_str, row_idx, col_idx = value_data

                    # Parse value - handle embedded flags if configured
                    if flag_in_result:
                        numeric_value, embedded_flag = parse_value_and_flag(value_str)
                    else:
                        numeric_value = parse_numeric_value(value_str)
                        embedded_flag = None

                    if numeric_value is not None:
                        # Extract unit from table if unit_column is specified
                        unit = field_config.get('unit', '')
                        unit_column = field_config.get('unit_column')
                        if unit_column is not None and unit_column < len(table.rows[row_idx]):
                            table_unit = table.rows[row_idx][unit_column].strip()
                            if table_unit:
                                unit = table_unit

                        # Extract reference range if available
                        ref_range = None
                        ref_column = field_config.get('ref_range_column', 4)
                        if ref_column < len(table.rows[row_idx]):
                            ref_str = table.rows[row_idx][ref_column]
                            ref_range = parse_reference_range(ref_str)

                        # Extract abnormal flag - use embedded flag or look in flag column
                        abnormal_flag = embedded_flag
                        if not abnormal_flag:
                            flag_column = field_config.get('flag_column')
                            if flag_column is not None:
                                abnormal_flag = extract_abnormal_flag(table, row_idx, flag_column)

                        # Get bounding box for the value cell
                        value_bbox = None
                        if hasattr(table, 'get_cell_bbox'):
                            # row_idx in template is 0-indexed within data rows
                            # Add 1 to account for header row in cell_boxes
                            value_bbox = table.get_cell_bbox(row_idx + 1, col_idx)

                        # Create extracted value
                        extracted = ExtractedValue(
                            field_name=field_name,
                            value=numeric_value,
                            unit=unit,
                            confidence=0.98,
                            extraction_method="template",
                            source_page=table.page_number,
                            source_location=f"Row {row_idx}, Column {col_idx}",
                            source_row_index=row_idx,  # For document order sorting
                            source_text=value_str,
                            bbox=value_bbox,
                            bbox_normalized=True,
                            reference_min=ref_range[0] if ref_range else None,
                            reference_max=ref_range[1] if ref_range else None,
                            abnormal_flag=abnormal_flag
                        )

                        context.add_extracted_value(extracted)
                        extracted_count += 1
                        found_fields.add(field_name)

        # Log extraction summary
        total_fields = len(template.get('field_mappings', {}))
        self.logger.info(
            f"Template extraction: {extracted_count}/{total_fields} fields extracted "
            f"from {len(tables)} tables"
        )

        # CHECK FOR SUSPICIOUS RESULTS: all values the same indicates column misalignment
        if extracted_count > 3:
            extracted_values = [v.value for v in context.extracted_values if v.value is not None]
            unique_values = set(extracted_values)
            if len(unique_values) <= 2:
                self.logger.warning(
                    f"SUSPICIOUS: {extracted_count} extractions but only {len(unique_values)} unique values "
                    f"({unique_values}). Likely column misalignment - trying MedGemma table parsing."
                )
                # Clear bad extractions
                context.extracted_values = []

                # Try MedGemma table parsing instead
                medgemma_results = await self._parse_tables_with_medgemma(tables)
                if medgemma_results and len(medgemma_results) >= 3:
                    for result in medgemma_results:
                        test_name = result.get('test_name', '')
                        if not test_name:
                            continue

                        # Try to find bbox for the value in the original tables
                        value_bbox = None
                        source_page = None
                        value_str = str(result.get('value', ''))
                        for table in tables:
                            if hasattr(table, 'find_value_bbox'):
                                value_bbox = table.find_value_bbox(value_str)
                                if value_bbox:
                                    source_page = table.page_number
                                    break

                        extracted = ExtractedValue(
                            field_name=test_name.lower().replace(' ', '_'),
                            value=result.get('value'),
                            unit=result.get('unit', ''),
                            confidence=0.85,
                            extraction_method="template_medgemma_recovery",
                            source_page=source_page,
                            bbox=value_bbox,
                            bbox_normalized=True if value_bbox else False,
                            reference_min=result.get('reference_min'),
                            reference_max=result.get('reference_max'),
                            abnormal_flag=result.get('abnormal_flag')
                        )
                        context.add_extracted_value(extracted)

                    return {
                        "decision": "extracted",
                        "confidence": 0.85,
                        "reasoning": f"Template parsing failed (column misalignment), MedGemma recovery: {len(context.extracted_values)} values",
                        "method": "template_medgemma_recovery",
                        "values_extracted": len(context.extracted_values),
                        "template_fields": total_fields
                    }

        confidence = 0.98 if extracted_count > 0 else 0.0

        return {
            "decision": "extracted",
            "confidence": confidence,
            "reasoning": f"Template extraction: {extracted_count}/{total_fields} values extracted",
            "method": "template",
            "values_extracted": extracted_count,
            "template_fields": total_fields
        }
    
    async def _extract_with_tables(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract using table detection (no template).

        Strategy:
        1. Use pdfplumber/PyMuPDF to extract table structure
        2. Send table data to MedGemma for intelligent parsing
        3. MedGemma handles column alignment issues that break programmatic parsing
        """
        self.logger.info("Extracting with table detection")

        # Use pre-extracted tables from parallel extraction (or extract if not available)
        tables = getattr(context, '_extracted_tables', None)
        if tables is None:
            tables = self.table_extractor.extract_tables(context.document_path)

        if not tables:
            return {
                "decision": "no_tables",
                "confidence": 0.0,
                "reasoning": "No tables found",
                "method": "table",
                "values_extracted": 0
            }

        # NEW: Send tables to MedGemma for intelligent parsing
        # This handles column misalignment issues that break programmatic parsing
        medgemma_result = await self._parse_tables_with_medgemma(tables)

        if medgemma_result and len(medgemma_result) >= 5:
            # MedGemma successfully parsed tables
            extracted_count = 0
            for idx, result in enumerate(medgemma_result):
                test_name = result.get('test_name', '')
                if not test_name:
                    continue

                # Try to find bbox for the value in the original tables
                value_bbox = None
                source_page = None
                value_str = str(result.get('value', ''))
                for table in tables:
                    if hasattr(table, 'find_value_bbox'):
                        value_bbox = table.find_value_bbox(value_str)
                        if value_bbox:
                            source_page = table.page_number
                            break

                extracted = ExtractedValue(
                    field_name=test_name.lower().replace(' ', '_'),
                    value=result.get('value'),
                    unit=result.get('unit', ''),
                    confidence=0.85,
                    extraction_method="table_medgemma",
                    source_page=source_page,
                    source_row_index=idx,  # For document order sorting
                    bbox=value_bbox,
                    bbox_normalized=True if value_bbox else False,
                    reference_min=result.get('reference_min'),
                    reference_max=result.get('reference_max'),
                    abnormal_flag=result.get('abnormal_flag')
                )
                context.add_extracted_value(extracted)
                extracted_count += 1

            if extracted_count > 0:
                self.logger.info(f"MedGemma table parsing: {extracted_count} values extracted")
                return {
                    "decision": "extracted",
                    "confidence": 0.85,
                    "reasoning": f"MedGemma table parsing: {extracted_count} values from {len(tables)} tables",
                    "method": "table_medgemma",
                    "values_extracted": extracted_count
                }

        # Fallback to programmatic parsing if MedGemma fails
        self.logger.info("MedGemma table parsing insufficient, trying programmatic parsing")

        extracted_count = 0

        # Process ALL tables, not just the largest one
        for table in tables:
            # Detect column indices from headers AND data patterns
            # Pass sample rows for pattern-based detection when headers are unclear
            col_indices = self._detect_column_indices(table.headers, table.rows[:10])

            for row_idx, row in enumerate(table.rows):
                if len(row) < 2:
                    continue

                # Get test name from first column (or detected test column)
                test_col = col_indices.get('test', 0)
                test_name = row[test_col].strip() if test_col < len(row) else ''

                # Skip empty test names or header-like rows
                if not test_name:
                    continue
                # Be more lenient - only skip if the entire row looks like a header
                # (e.g., "Test | Result | Unit" pattern)
                skip_keywords = ['result', 'panel', 'ordered', 'component', 'biological ref']
                if test_name.lower() in ['test', 'test name', 'analyte'] and row_idx < 2:
                    continue
                if any(test_name.lower() == skip for skip in skip_keywords):
                    continue

                # Get value from result column
                result_col = col_indices.get('result', 1)
                value_str = row[result_col].strip() if result_col < len(row) else ''

                # Skip rows without values
                if not value_str or value_str.lower() in ['result', 'value', '']:
                    continue

                # If result column returns non-numeric, try adjacent columns
                # This handles cases where method columns are between test and value
                numeric_value, embedded_flag = parse_value_and_flag(value_str)

                if numeric_value is None and result_col < len(row) - 1:
                    # Try the next column (common in formats with method column)
                    for alt_col in range(result_col + 1, min(result_col + 3, len(row))):
                        alt_value_str = row[alt_col].strip()
                        alt_numeric, alt_flag = parse_value_and_flag(alt_value_str)
                        if alt_numeric is not None:
                            value_str = alt_value_str
                            numeric_value = alt_numeric
                            embedded_flag = alt_flag
                            self.logger.debug(
                                f"Found numeric in alt column {alt_col} for {test_name}: {numeric_value}"
                            )
                            break

                if numeric_value is not None:
                    # Track which column we actually found the value in
                    actual_value_col = result_col
                    for check_col in range(result_col, min(result_col + 3, len(row))):
                        if row[check_col].strip() == value_str:
                            actual_value_col = check_col
                            break

                    # Extract flag - use embedded flag first, then check flag column or adjacent columns
                    abnormal_flag = embedded_flag
                    if not abnormal_flag:
                        flag_col = col_indices.get('flag')
                        if flag_col is not None and flag_col < len(row):
                            flag_str = row[flag_col].strip().upper()
                            if flag_str in ['H', 'L', 'HH', 'LL', 'HIGH', 'LOW', 'CRITICAL', '*']:
                                abnormal_flag = flag_str
                        # Also check column before value for flags (Sterling Accuris format)
                        if not abnormal_flag and actual_value_col > 0:
                            flag_str = row[actual_value_col - 1].strip().upper()
                            if flag_str in ['H', 'L', 'HH', 'LL', 'HIGH', 'LOW', 'CRITICAL', '*']:
                                abnormal_flag = flag_str

                    # Extract unit - check detected column or column after value
                    unit_col = col_indices.get('unit')
                    unit = ''
                    if unit_col is not None and unit_col < len(row):
                        unit = row[unit_col].strip()
                    # If no unit found, check column after the actual value column
                    if not unit and actual_value_col + 1 < len(row):
                        potential_unit = row[actual_value_col + 1].strip()
                        # Verify it looks like a unit (contains unit-like patterns)
                        import re
                        if re.search(r'(g/dL|mg/dL|mmol|/cmm|/uL|fL|pg|%|IU|ng|µg|micro|mm/)', potential_unit, re.IGNORECASE):
                            unit = potential_unit

                    # Extract reference range
                    ref_col = col_indices.get('reference')
                    ref_range = None
                    if ref_col is not None and ref_col < len(row):
                        ref_range = parse_reference_range(row[ref_col])
                    # Also check last column for reference ranges
                    if ref_range is None and len(row) > 2:
                        ref_range = parse_reference_range(row[-1])

                    # Get bounding box for the value cell
                    value_bbox = None
                    if hasattr(table, 'get_cell_bbox'):
                        # row_idx is 0-indexed within data rows, add 1 for header
                        value_bbox = table.get_cell_bbox(row_idx + 1, actual_value_col)

                    # Create extracted value
                    extracted = ExtractedValue(
                        field_name=test_name.lower().replace(' ', '_'),
                        value=numeric_value,
                        unit=unit,
                        confidence=0.85,
                        extraction_method="table",
                        source_page=table.page_number,
                        source_location=f"Row {row_idx}",
                        source_row_index=row_idx,  # For document order sorting
                        source_text=value_str,
                        bbox=value_bbox,
                        bbox_normalized=True,
                        reference_min=ref_range[0] if ref_range else None,
                        reference_max=ref_range[1] if ref_range else None,
                        abnormal_flag=abnormal_flag
                    )

                    context.add_extracted_value(extracted)
                    extracted_count += 1

        confidence = 0.85 if extracted_count > 0 else 0.0

        return {
            "decision": "extracted",
            "confidence": confidence,
            "reasoning": f"Table extraction: {extracted_count} values from {len(tables)} tables",
            "method": "table",
            "values_extracted": extracted_count
        }

    def _detect_column_indices(self, headers: list, sample_rows: list = None) -> dict:
        """
        Detect column indices from table headers and data patterns.

        Common lab report formats:
        - LabCorp: TESTS | RESULT | FLAG | UNITS | REFERENCE INTERVAL | LAB
        - Quest: Test Name | Result | Flag | Units | Reference Range
        - Generic: Test | Value | Unit | Reference | Flag
        - Sterling Accuris (India): Test | Method | Flag | Value | Unit | Reference

        Uses both header keywords AND data pattern analysis to handle formats
        where headers are missing or where there's a "method" column between
        test name and value.

        Returns dict with keys: test, result, flag, unit, reference
        """
        indices = {
            'test': 0,      # Default first column
            'result': 1,    # Default second column
            'flag': None,
            'unit': None,
            'reference': None
        }

        if not headers:
            # No headers - try to detect from data patterns
            if sample_rows:
                return self._detect_columns_from_data(sample_rows)
            return indices

        headers_lower = [h.lower() for h in headers]
        header_found_result = False

        for i, header in enumerate(headers_lower):
            # Test name column
            if any(kw in header for kw in ['test', 'analyte', 'component', 'name']):
                indices['test'] = i
            # Result/Value column
            elif any(kw in header for kw in ['result', 'value', 'finding']):
                indices['result'] = i
                header_found_result = True
            # Flag column
            elif any(kw in header for kw in ['flag', 'abnormal', 'status', 'interp']):
                indices['flag'] = i
            # Unit column
            elif any(kw in header for kw in ['unit', 'uom']):
                indices['unit'] = i
            # Reference range column
            elif any(kw in header for kw in ['reference', 'range', 'normal', 'interval']):
                indices['reference'] = i

        # If result column wasn't found in headers, use data pattern detection
        if not header_found_result and sample_rows:
            data_indices = self._detect_columns_from_data(sample_rows)
            # Only override result column if we found a better one
            if data_indices.get('result') and data_indices['result'] != 1:
                indices['result'] = data_indices['result']
                self.logger.debug(f"Overriding result column from data patterns: col {indices['result']}")
            # Also pick up flag, unit, reference if detected
            for key in ['flag', 'unit', 'reference']:
                if data_indices.get(key) is not None and indices.get(key) is None:
                    indices[key] = data_indices[key]

        self.logger.debug(f"Detected column indices from headers {headers}: {indices}")
        return indices

    def _detect_columns_from_data(self, sample_rows: list) -> dict:
        """
        Detect column types from actual data patterns.

        Analyzes sample rows to find:
        - Numeric columns (likely result values)
        - Flag columns (H/L patterns)
        - Unit columns (g/dL, mg/dL, etc.)
        - Reference range columns (X-Y patterns)

        This handles formats like Sterling Accuris where there's a "method"
        column between test name and result.
        """
        import re

        indices = {
            'test': 0,
            'result': None,
            'flag': None,
            'unit': None,
            'reference': None
        }

        if not sample_rows:
            return indices

        # Patterns for detecting column types
        numeric_pattern = re.compile(r'^[<>]?\s*\d+\.?\d*$')
        flag_pattern = re.compile(r'^[HL]$|^(High|Low)$', re.IGNORECASE)
        unit_pattern = re.compile(r'(g/dL|mg/dL|mmol/L|/cmm|/uL|/mm|fL|pg|%|IU/mL|ng/mL|µg/dL|micro\s*g)', re.IGNORECASE)
        ref_range_pattern = re.compile(r'\d+\.?\d*\s*-\s*\d+\.?\d*')

        # Count matches per column
        max_cols = max(len(row) for row in sample_rows) if sample_rows else 0
        numeric_counts = [0] * max_cols
        flag_counts = [0] * max_cols
        unit_counts = [0] * max_cols
        ref_counts = [0] * max_cols

        for row in sample_rows[:10]:  # Analyze up to 10 rows
            for i, cell in enumerate(row):
                cell_str = str(cell).strip()
                if not cell_str:
                    continue

                if numeric_pattern.match(cell_str):
                    numeric_counts[i] += 1
                if flag_pattern.match(cell_str):
                    flag_counts[i] += 1
                if unit_pattern.search(cell_str):
                    unit_counts[i] += 1
                if ref_range_pattern.search(cell_str):
                    ref_counts[i] += 1

        # Find best column for each type
        # Result column: highest numeric count (excluding column 0 which is usually test name)
        if max_cols > 1:
            numeric_candidates = [(i, count) for i, count in enumerate(numeric_counts) if i > 0]
            if numeric_candidates:
                best_numeric = max(numeric_candidates, key=lambda x: x[1])
                if best_numeric[1] >= 2:  # At least 2 numeric values
                    indices['result'] = best_numeric[0]

        # Flag column: highest flag count
        if max(flag_counts) >= 1:
            indices['flag'] = flag_counts.index(max(flag_counts))

        # Unit column: highest unit count
        if max(unit_counts) >= 2:
            indices['unit'] = unit_counts.index(max(unit_counts))

        # Reference column: highest reference range count
        if max(ref_counts) >= 2:
            indices['reference'] = ref_counts.index(max(ref_counts))

        self.logger.debug(
            f"Detected columns from data patterns: result={indices['result']}, "
            f"flag={indices['flag']}, unit={indices['unit']}, ref={indices['reference']} "
            f"(numeric_counts={numeric_counts[:6]}, flag_counts={flag_counts[:6]})"
        )

        return indices
    
    async def _extract_with_medgemma(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract using MedGemma (fallback for unknown formats).

        Uses AI to understand unstructured lab reports.
        Also generates a draft template for future use.
        """
        self.logger.info("Extracting with MedGemma (full extraction mode)")

        # Get text from PDF
        text = context.raw_text or self.text_extractor.extract_text(context.document_path)

        # Log text length for debugging
        self.logger.info(f"Raw text length: {len(text)} chars")

        # Use the shared extraction method with improved prompt (multi-page aware)
        lab_results = await self._run_medgemma_extraction(text, context.total_pages or 1)

        # Trigger auto-template generation in background
        try:
            await self._generate_draft_template(context, text)
        except Exception as e:
            self.logger.warning(f"Auto-template generation failed (non-blocking): {e}")

        # If we got results from the shared method, convert to ExtractedValues
        if lab_results and isinstance(lab_results, list):
            extracted_count = 0
            for idx, result in enumerate(lab_results):
                test_name = result.get('test_name', '')
                if not test_name:
                    continue

                field_name = test_name.lower().replace(' ', '_')
                extracted = ExtractedValue(
                    field_name=field_name,
                    value=result.get('value'),
                    unit=result.get('unit', ''),
                    confidence=calculate_extraction_confidence(
                        value=result.get('value'),
                        unit=result.get('unit', ''),
                        reference_min=result.get('reference_min'),
                        reference_max=result.get('reference_max'),
                        extraction_method="medgemma",
                        field_name=field_name
                    ),
                    extraction_method="medgemma",
                    source_row_index=idx,  # For document order sorting
                    reference_min=result.get('reference_min'),
                    reference_max=result.get('reference_max'),
                    abnormal_flag=result.get('abnormal_flag')
                )
                context.add_extracted_value(extracted)
                extracted_count += 1

            self.logger.info(f"MedGemma extracted {extracted_count} values")

            return {
                "decision": "extracted",
                "confidence": 0.75 if extracted_count > 0 else 0.0,
                "reasoning": f"MedGemma extraction: {extracted_count} values",
                "method": "medgemma",
                "values_extracted": extracted_count
            }

        # Fallback to original method if shared extraction fails
        self.logger.warning("Shared MedGemma extraction failed, trying direct approach")

        # Use more text - up to 12000 chars
        text_sample = text[:12000]

        prompt = f"""Extract ALL lab test results from this medical document as a JSON object.

Document text:
{text_sample}

Return a JSON object with a "results" array containing EVERY lab test found:
{{
  "results": [
    {{"test_name": "Hemoglobin", "value": 12.5, "unit": "g/dL", "reference_min": 12.0, "reference_max": 15.5, "abnormal_flag": null}},
    {{"test_name": "WBC", "value": 7.5, "unit": "x10E3/uL", "reference_min": 4.5, "reference_max": 11.0, "abnormal_flag": null}},
    ... include ALL tests found in the document ...
  ]
}}

IMPORTANT: Extract EVERY lab value from the document. There may be 20-50 results.
Use null for any missing fields. Return ONLY the JSON object, no other text.

JSON:"""

        try:
            # Use json_mode=True to ensure valid JSON output from Ollama
            # Increase max_tokens to accommodate many lab results
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.1,
                json_mode=True
            )

            result_text = response.get('text', '')

            # Parse JSON - handle multiple formats with repair fallback
            lab_results = None
            json_repaired = False  # Track if json_repair was used

            def extract_results_from_parsed(parsed):
                """Extract lab results from parsed JSON (array or dict)."""
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    # Check if it's a single lab result (has test_name key)
                    if 'test_name' in parsed:
                        return [parsed]
                    # Check common keys for array of results
                    nested = (
                        parsed.get('results') or
                        parsed.get('lab_results') or
                        parsed.get('data') or
                        parsed.get('values')
                    )
                    if isinstance(nested, list):
                        return nested
                    elif isinstance(nested, dict) and 'test_name' in nested:
                        return [nested]
                return None

            # Try 1: Parse entire response as JSON
            try:
                parsed = json.loads(result_text)
                lab_results = extract_results_from_parsed(parsed)
                if lab_results:
                    self.logger.debug(f"Parsed JSON directly: {len(lab_results)} results")
            except json.JSONDecodeError:
                pass

            # Try 2: Use json_repair to fix malformed JSON
            if not lab_results:
                try:
                    repaired = repair_json(result_text, return_objects=True)
                    lab_results = extract_results_from_parsed(repaired)
                    if lab_results:
                        json_repaired = True
                        self.logger.warning(
                            f"json_repair fixed malformed JSON: {len(lab_results)} results. "
                            f"Original response (first 200 chars): {result_text[:200]}"
                        )
                except Exception as e:
                    self.logger.debug(f"json_repair failed: {e}")

            # Try 3: Extract JSON array with regex
            if not lab_results:
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    try:
                        lab_results = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # Try repair on extracted array
                        try:
                            repaired = repair_json(json_match.group(), return_objects=True)
                            if isinstance(repaired, list):
                                lab_results = repaired
                                json_repaired = True
                                self.logger.warning(
                                    f"json_repair fixed extracted array: {len(lab_results)} results. "
                                    f"Original array (first 200 chars): {json_match.group()[:200]}"
                                )
                        except Exception:
                            pass

            # Try 4: Extract single JSON object with regex (if MedGemma returned one result)
            if not lab_results:
                json_match = re.search(r'\{[^{}]*"test_name"[^{}]*\}', result_text, re.DOTALL)
                if json_match:
                    try:
                        single_result = json.loads(json_match.group())
                        lab_results = [single_result]
                        self.logger.debug("Extracted single result via regex")
                    except json.JSONDecodeError:
                        try:
                            repaired = repair_json(json_match.group(), return_objects=True)
                            if isinstance(repaired, dict):
                                lab_results = [repaired]
                                json_repaired = True
                                self.logger.warning(
                                    f"json_repair fixed single result. "
                                    f"Original: {json_match.group()[:200]}"
                                )
                        except Exception:
                            pass

            if lab_results and isinstance(lab_results, list):
                extracted_count = 0

                # Determine confidence based on whether json_repair was used
                base_confidence = 0.75
                if json_repaired:
                    base_confidence -= JSON_REPAIR_CONFIDENCE_PENALTY
                    self.logger.warning(
                        f"JSON repair was used - confidence reduced to {base_confidence:.2f}. "
                        "Some data may have been silently corrected or lost."
                    )
                    # Add warning for human review
                    context.warnings.append(
                        "MedGemma output required JSON repair - verify extracted values"
                    )
                    context.requires_review = True
                    context.review_reasons.append(
                        "JSON repair was used during extraction - potential data loss"
                    )

                for result in lab_results:
                    extracted = ExtractedValue(
                        field_name=result.get('test_name', '').lower().replace(' ', '_'),
                        value=result.get('value'),
                        unit=result.get('unit', ''),
                        confidence=base_confidence,
                        extraction_method="medgemma" + ("_repaired" if json_repaired else ""),
                        reference_min=result.get('reference_min'),
                        reference_max=result.get('reference_max'),
                        abnormal_flag=result.get('abnormal_flag')
                    )

                    context.add_extracted_value(extracted)
                    extracted_count += 1

                return {
                    "decision": "extracted",
                    "confidence": base_confidence,
                    "reasoning": f"MedGemma extraction: {extracted_count} values" +
                                 (" (JSON repaired)" if json_repaired else ""),
                    "method": "medgemma",
                    "values_extracted": extracted_count,
                    "json_repaired": json_repaired
                }

        except Exception as e:
            self.logger.error(f"MedGemma extraction failed: {e}")

        return {
            "decision": "extraction_failed",
            "confidence": 0.0,
            "reasoning": "MedGemma extraction failed",
            "method": "medgemma",
            "values_extracted": 0
        }

    async def _generate_draft_template(self, context: ProcessingContext, text: str) -> None:
        """
        Generate a draft template for unknown document format.

        This enables the system to learn new formats automatically.
        Templates are saved as drafts for admin review.
        """
        try:
            from ...template_generator import TemplateGenerator
        except ImportError:
            self.logger.debug("TemplateGenerator not available - skipping draft template generation")
            return

        self.logger.info("Generating draft template for unknown lab format...")

        generator = TemplateGenerator(self.config)

        # Get visual identification if available
        visual_info = context.sections.get('visual_identification')

        result = await generator.generate_template(
            pdf_path=context.document_path,
            document_type='lab',
            extracted_text=text,
            visual_info=visual_info
        )

        if result.get('draft_path'):
            self.logger.info(
                f"Draft template generated: {result['draft_path']} "
                f"(confidence: {result.get('confidence', 0):.2f})"
            )

            # Store template info in context for audit trail
            context.sections['auto_template'] = {
                'draft_path': str(result['draft_path']),
                'template_id': result.get('template', {}).get('id'),
                'confidence': result.get('confidence', 0),
                'needs_review': True,
                'validation_notes': result.get('validation_notes', [])
            }

            # Flag for review
            context.requires_review = True
            context.review_reasons.append(
                "Auto-generated template requires admin approval before production use"
            )