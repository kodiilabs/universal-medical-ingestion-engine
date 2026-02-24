# ============================================================================
# src/medical_ingestion/extractors/azure_openai_client.py
# ============================================================================
"""
Azure OpenAI Vision Client for Medical Document Extraction

When USE_CLOUD=True, this client replaces local VLM + OCR with Azure GPT-4o vision.
Returns data in the same format as ContentAgnosticExtractor for seamless integration.

Key features:
- Single API call handles both OCR and extraction (GPT-4o vision)
- Matches GenericMedicalExtraction return format
- Supports PDF and image files
- Falls back gracefully on errors

Usage:
    from medical_ingestion.extractors.azure_openai_client import AzureOpenAIExtractor

    extractor = AzureOpenAIExtractor(config)
    result = await extractor.extract_from_file(Path("document.pdf"))
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Import data classes from content_agnostic_extractor
from .content_agnostic_extractor import (
    GenericMedicalExtraction,
    PatientInfo,
    TestResult,
    MedicationInfo,
    ClinicalFinding,
    ProcedureInfo,
    DateInfo,
    ProviderInfo,
    OrganizationInfo,
)


# ── Shared prompt fragments ──────────────────────────────────────────────────

_PROMPT_HEADER = """Analyze this medical document image. You must do TWO things in a single JSON response:

1. CLASSIFY the document type
2. EXTRACT all data from the document

Return JSON with ALL of the following fields:
"""

_PROMPT_HEADER_BBOX = """Analyze this medical document image. You must do THREE things in a single JSON response:

1. CLASSIFY the document type
2. EXTRACT all data from the document
3. Return BOUNDING BOXES for every extracted field

Return JSON with ALL of the following fields:
"""

_PROMPT_SCHEMA = """{
  "document_classification": {
    "document_type": "<one of: lab, radiology, pathology, prescription, insurance, clinical_notes, discharge_summary, referral, consent, billing, receipt, dental, vision, or a descriptive type>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<brief explanation of why this type was chosen>",
    "insurance_classification": {
      "submission_type": "<claim | predetermination | null if not insurance>",
      "claim_type": "<drug | dental | extended_health_care | vision | travel | null>",
      "line_of_benefits": "<specific benefit line e.g. paramedical_services, dental_services, prescription_drugs, vision_care, medical_supplies, travel_emergency, null>",
      "benefit_type": "<specific benefit e.g. physiotherapy, major_restorative, preventive, null>"
    }
  },
  "patient": {"name": "full name", "dob": "birth date only", "gender": "M/F", "mrn": "medical record #", "address": "address", "phone": "phone", "insurance": "insurance ID or policy #"%%PATIENT_BBOX%%},
  "test_results": [{"name": "test name", "value": "result", "unit": "unit", "reference_range": "range", "abnormal_flag": "H/L/null"%%ITEM_BBOX%%}],
  "medications": [{"name": "drug name + strength", "frequency": "dosing", "instructions": "sig"%%ITEM_BBOX%%}],
  "findings": [{"finding": "diagnosis or impression", "category": "diagnosis/impression/recommendation"%%ITEM_BBOX%%}],
  "procedures": [{"procedure": "procedure name", "code": "procedure code if shown", "tooth": "tooth # if dental", "surface": "surface if dental", "date": "date of service"%%ITEM_BBOX%%}],
  "dates": [{"date_type": "service/collection/report/submission", "date_value": "the date"%%ITEM_BBOX%%}],
  "providers": [{"name": "provider name", "role": "role/title", "npi": "NPI if shown", "license": "license # if shown"%%ITEM_BBOX%%}],
  "organizations": [{"name": "organization", "address": "address", "phone": "phone", "type": "clinic/lab/insurer/pharmacy"%%ITEM_BBOX%%}],
  "financial": {"invoice_number": "invoice #", "total": "total amount", "currency": "$", "amounts": [{"description": "line item", "amount": "dollar amount"%%ITEM_BBOX%%}]},
  "insurance_details": {
    "carrier": "insurance company name",
    "group_number": "group #",
    "policy_number": "policy/certificate #",
    "member_id": "member/participant ID",
    "plan_type": "plan description",
    "subscriber": "policyholder name if different from patient",
    "relationship": "relationship to subscriber"
  }
}
"""

_BBOX_RULES = """
BOUNDING BOX RULES:
- "bbox" is [x, y, width, height] as PERCENTAGES (0-100) of the image dimensions
- x = left edge %, y = top edge %, width = box width %, height = box height %
- Estimate the bounding box around the VALUE portion of each field (not the label)
- For test results: draw the box around the entire row (name + value + unit + range)
- For patient info: draw one box around the patient information block
- Include bbox for EVERY extracted item - this is critical for document overlay display
- If you cannot determine a bounding box for an item, omit the bbox field for that item
- For multi-page documents: add "page": <1-based page number> to each item to indicate which page it appears on
"""

_CLASSIFICATION_RULES = """
CLASSIFICATION RULES:
- Look at the ENTIRE document including headers, logos, form titles, and watermarks
- For insurance documents, ALWAYS fill insurance_classification:
  - submission_type: "claim" (already received service) vs "predetermination" (requesting pre-approval)
  - claim_type: "drug" (prescriptions), "dental" (dental procedures), "extended_health_care" (paramedical, medical supplies), "vision" (eye care), "travel" (travel emergency)
  - line_of_benefits: the specific benefit category (e.g. "paramedical_services", "dental_services", "prescription_drugs", "vision_care", "medical_supplies")
  - benefit_type: the specific service (e.g. "physiotherapy", "major_restorative", "preventive", "glasses_contacts")
- If NOT an insurance document, set insurance_classification fields to null

EXTRACTION RULES:
- Return null or [] if data doesn't exist - don't invent values
- CRITICAL: Extract EVERY piece of data visible on the document
- For lab reports: include ALL test result rows (WBC, RBC, Hemoglobin, Platelets, differentials, etc.)
- For prescriptions: include ALL medications with dosing instructions
- For radiology: include ALL findings and impressions
- For dental: include ALL procedures with tooth numbers and surfaces
- For insurance claims: include ALL line items, amounts, procedure codes, and policy details
- Preserve exact values as shown in the document
- Do NOT include duplicate entries — if the same test/medication/finding appears multiple times, include it ONLY ONCE

Return ONLY valid JSON, no explanations or markdown:"""


def _build_extraction_prompt(include_bbox: bool) -> str:
    """Build the extraction prompt, optionally including bounding box instructions."""
    header = _PROMPT_HEADER_BBOX if include_bbox else _PROMPT_HEADER
    bbox_suffix = ', "bbox": [x, y, w, h]' if include_bbox else ''
    schema = _PROMPT_SCHEMA.replace('%%PATIENT_BBOX%%', bbox_suffix).replace('%%ITEM_BBOX%%', bbox_suffix)
    rules = (_BBOX_RULES + _CLASSIFICATION_RULES) if include_bbox else _CLASSIFICATION_RULES
    return header + schema + rules


# Pre-built prompts
AZURE_EXTRACTION_PROMPT = _build_extraction_prompt(include_bbox=False)
AZURE_EXTRACTION_PROMPT_BBOX = _build_extraction_prompt(include_bbox=True)


class AzureOpenAIExtractor:
    """
    Azure OpenAI GPT-4o vision-based medical document extractor.

    Replaces local VLM + OCR when USE_CLOUD=True.
    Returns GenericMedicalExtraction for compatibility with existing pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._client = None

        # Azure configuration (from config or environment)
        import os
        self.azure_endpoint = self.config.get('azure_endpoint') or os.getenv('AZURE_OPENAI_ENDPOINT', '')
        self.azure_api_key = self.config.get('azure_api_key') or os.getenv('AZURE_OPENAI_API_KEY', '')
        self.azure_deployment = self.config.get('azure_deployment') or os.getenv('AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT', 'gpt-4o')
        self.azure_api_version = self.config.get('azure_api_version') or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')

        # Bounding boxes flag (controlled by RETURN_BOUNDING_BOXES env var)
        self.return_bounding_boxes = self.config.get(
            'return_bounding_boxes',
            os.getenv('RETURN_BOUNDING_BOXES', 'false').lower() in ('true', '1', 'yes')
        )

        # Generation settings — need more tokens when bbox is on
        default_tokens = 16000 if self.return_bounding_boxes else 4000
        self.max_tokens = self.config.get('max_tokens', default_tokens)
        self.temperature = self.config.get('temperature', 0.1)

        # Validate configuration
        if not self.azure_endpoint or not self.azure_api_key:
            logger.warning("Azure OpenAI credentials not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")

        logger.info(f"Azure extractor: bounding_boxes={'ON' if self.return_bounding_boxes else 'OFF'}, max_tokens={self.max_tokens}")

    @property
    def client(self):
        """Lazy load Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version
                )
                logger.info(f"Azure OpenAI client initialized: endpoint={self.azure_endpoint}, deployment={self.azure_deployment}")
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                raise
        return self._client

    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        return bool(self.azure_endpoint and self.azure_api_key and self.azure_deployment)

    async def extract_from_file(
        self,
        file_path: Path,
        extraction_hints: Dict[str, Any] = None
    ) -> GenericMedicalExtraction:
        """
        Extract medical data from a document file using GPT-4o vision.

        Args:
            file_path: Path to PDF or image file
            extraction_hints: Optional hints from similar documents

        Returns:
            GenericMedicalExtraction with all extracted data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return GenericMedicalExtraction(warnings=[f"File not found: {file_path}"])

        if not self.is_configured():
            logger.error("Azure OpenAI not configured")
            return GenericMedicalExtraction(warnings=["Azure OpenAI credentials not configured"])

        try:
            import time as _time

            # Phase 1: Convert document to images for GPT-4o vision
            t0 = _time.time()
            images = await self._document_to_images(file_path)
            conversion_time = _time.time() - t0

            if not images:
                logger.error(f"Could not convert document to images: {file_path}")
                return GenericMedicalExtraction(warnings=["Could not process document"])

            # Phase 2: Extract from images using GPT-4o
            logger.info(f"Extracting from {len(images)} page(s) using Azure GPT-4o vision")
            t1 = _time.time()
            llm_result = await self._extract_from_images(images)
            api_call_time = _time.time() - t1

            # Phase 3: Structure the response into GenericMedicalExtraction
            t2 = _time.time()
            extraction = self._structure_extraction(llm_result)
            structuring_time = _time.time() - t2

            extraction.extraction_method = "azure_gpt4o_vision"

            # Embed timing and document metadata for the pipeline to use
            extraction.raw_fields['_extraction_metadata'] = {
                "page_count": len(images),
                "file_type": file_path.suffix.lower(),
                "file_size_kb": round(file_path.stat().st_size / 1024, 1),
                "conversion_time": round(conversion_time, 3),
                "api_call_time": round(api_call_time, 3),
                "structuring_time": round(structuring_time, 3),
                "total_time": round(conversion_time + api_call_time + structuring_time, 3),
                "model": self.azure_deployment,
                "endpoint": self.azure_endpoint,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            return extraction

        except Exception as e:
            logger.error(f"Azure extraction failed: {type(e).__name__}: {e}")
            return GenericMedicalExtraction(
                warnings=[f"Azure extraction failed: {e}"],
                extraction_method="azure_gpt4o_vision_failed"
            )

    async def extract_from_text(
        self,
        text: str,
        extraction_hints: Dict[str, Any] = None
    ) -> GenericMedicalExtraction:
        """
        Extract medical data from text using GPT-4o (text-only mode).

        This is used when text is already available (e.g., from local OCR).
        Falls back to text extraction if vision is not needed.

        Args:
            text: Extracted document text
            extraction_hints: Optional hints from similar documents

        Returns:
            GenericMedicalExtraction with all extracted data
        """
        if not text or len(text.strip()) < 50:
            return GenericMedicalExtraction(warnings=["Text too short for extraction"])

        if not self.is_configured():
            return GenericMedicalExtraction(warnings=["Azure OpenAI credentials not configured"])

        try:
            # Use text-based extraction (no vision)
            llm_result = await self._extract_from_text(text)

            # Structure the response
            extraction = self._structure_extraction(llm_result)
            extraction.extraction_method = "azure_gpt4o_text"

            return extraction

        except Exception as e:
            logger.error(f"Azure text extraction failed: {e}")
            return GenericMedicalExtraction(
                warnings=[f"Azure extraction failed: {e}"],
                extraction_method="azure_gpt4o_text_failed"
            )

    async def _document_to_images(self, file_path: Path) -> List[bytes]:
        """
        Convert document to list of images for GPT-4o vision.

        Returns:
            List of image bytes (PNG format)
        """
        suffix = file_path.suffix.lower()

        if suffix in ('.png', '.jpg', '.jpeg', '.gif', '.webp'):
            # Already an image - read directly
            with open(file_path, 'rb') as f:
                return [f.read()]

        elif suffix == '.pdf':
            # Convert PDF pages to images
            try:
                import pymupdf

                doc = pymupdf.open(file_path)
                images = []

                # Process each page (limit to first 10 pages for large documents)
                max_pages = min(len(doc), 10)

                for page_num in range(max_pages):
                    page = doc[page_num]
                    # Render at 150 DPI for good quality while keeping size reasonable
                    mat = pymupdf.Matrix(150/72, 150/72)
                    pix = page.get_pixmap(matrix=mat)
                    images.append(pix.tobytes("png"))

                doc.close()
                logger.info(f"Converted {len(images)} PDF pages to images")
                return images

            except ImportError:
                logger.error("pymupdf required for PDF processing")
                return []
            except Exception as e:
                logger.error(f"PDF conversion failed: {e}")
                return []

        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return []

    async def _extract_from_images(self, images: List[bytes]) -> Dict[str, Any]:
        """
        Extract data from images using GPT-4o vision API.

        Args:
            images: List of image bytes (PNG format)

        Returns:
            Parsed JSON response from GPT-4o
        """
        import asyncio

        # Build message content with all images
        prompt = AZURE_EXTRACTION_PROMPT_BBOX if self.return_bounding_boxes else AZURE_EXTRACTION_PROMPT
        content = [{"type": "text", "text": prompt}]

        for i, image_bytes in enumerate(images):
            # Encode image as base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"  # Use high detail for medical documents
                }
            })

        # Make API call (run in executor since openai client is sync)
        def call_api():
            response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content

        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, call_api)

        # Parse JSON response
        return self._parse_json(response_text)

    async def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract data from text using GPT-4o (text-only, no vision).

        Args:
            text: Document text

        Returns:
            Parsed JSON response from GPT-4o
        """
        import asyncio

        # Use same prompt format but with text instead of image
        prompt = f"""Analyze this medical document. You must do TWO things in a single JSON response:

1. CLASSIFY the document type
2. EXTRACT all data from the document

<DOCUMENT>
{text}
</DOCUMENT>

Return JSON with ALL of the following fields:

{{
  "document_classification": {{
    "document_type": "<one of: lab, radiology, pathology, prescription, insurance, clinical_notes, discharge_summary, referral, consent, billing, receipt, dental, vision, or a descriptive type>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<brief explanation>",
    "insurance_classification": {{
      "submission_type": "<claim | predetermination | null if not insurance>",
      "claim_type": "<drug | dental | extended_health_care | vision | travel | null>",
      "line_of_benefits": "<specific benefit line or null>",
      "benefit_type": "<specific benefit or null>"
    }}
  }},
  "patient": {{"name": "full name", "dob": "birth date only", "gender": "M/F", "mrn": "medical record #", "address": "address", "phone": "phone", "insurance": "insurance ID or policy #"}},
  "test_results": [{{"name": "test name", "value": "result", "unit": "unit", "reference_range": "range", "abnormal_flag": "H/L/null"}}],
  "medications": [{{"name": "drug name + strength", "frequency": "dosing", "instructions": "sig"}}],
  "findings": [{{"finding": "diagnosis or impression", "category": "diagnosis/impression/recommendation"}}],
  "procedures": [{{"procedure": "procedure name", "code": "procedure code if shown", "tooth": "tooth # if dental", "surface": "surface if dental", "date": "date of service"}}],
  "dates": [{{"date_type": "service/collection/report/submission", "date_value": "the date"}}],
  "providers": [{{"name": "provider name", "role": "role/title", "npi": "NPI if shown", "license": "license # if shown"}}],
  "organizations": [{{"name": "organization", "address": "address", "phone": "phone", "type": "clinic/lab/insurer/pharmacy"}}],
  "financial": {{"invoice_number": "invoice #", "total": "total amount", "currency": "$", "amounts": [{{"description": "line item", "amount": "dollar amount"}}]}},
  "insurance_details": {{
    "carrier": "insurance company name",
    "group_number": "group #",
    "policy_number": "policy/certificate #",
    "member_id": "member/participant ID",
    "plan_type": "plan description",
    "subscriber": "policyholder name if different from patient",
    "relationship": "relationship to subscriber"
  }}
}}

Rules:
- Return null or [] if data doesn't exist - don't invent values
- For insurance documents, ALWAYS fill insurance_classification with submission_type, claim_type, line_of_benefits, benefit_type
- Extract EVERY piece of data from the document

Return ONLY valid JSON:"""

        def call_api():
            response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content

        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, call_api)

        return self._parse_json(response_text)

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with error handling."""
        if not response:
            return {}

        response = response.strip()

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Look for JSON object in response
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
        return {}

    @staticmethod
    def _parse_bbox(item: Dict[str, Any], default_page: int = 1) -> Optional[Dict[str, Any]]:
        """
        Parse a bbox field from a GPT-4o item into frontend format.

        GPT-4o returns bbox as [x, y, width, height] in percentages (0-100).
        Frontend expects {x, y, width, height, page} in the same format.

        Returns None if bbox is missing or invalid.
        """
        bbox = item.get('bbox')
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        try:
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            # Sanity check: values should be 0-100 percentages
            if all(0 <= v <= 100 for v in [x, y, w, h]) and w > 0 and h > 0:
                page = int(item.get('page', default_page))
                return {"x": x, "y": y, "width": w, "height": h, "page": page}
        except (ValueError, TypeError):
            pass

        return None

    def _structure_extraction(self, llm_result: Dict[str, Any]) -> GenericMedicalExtraction:
        """
        Convert GPT-4o output to GenericMedicalExtraction.

        This mirrors the _structure_extraction method in ContentAgnosticExtractor
        to ensure consistent output format. Also extracts document_classification
        which is stored in raw_fields for the pipeline to use.

        When return_bounding_boxes is enabled, bounding boxes from GPT-4o are
        collected into raw_fields['_bounding_boxes'] as a flat list matching
        the frontend's expected format:
        [{field, value, x, y, width, height, page, confidence}, ...]
        """
        warnings = []
        collect_bbox = self.return_bounding_boxes
        bounding_boxes = [] if collect_bbox else None  # Only allocate when needed

        if not llm_result:
            warnings.append("GPT-4o returned empty response")
            return GenericMedicalExtraction(warnings=warnings)

        # Parse patient info
        patient = None
        patient_data = llm_result.get('patient', {})
        if patient_data and isinstance(patient_data, dict):
            patient_fields = ['name', 'dob', 'gender', 'mrn', 'address', 'phone', 'insurance']
            filled = sum(1 for f in patient_fields if patient_data.get(f))
            confidence = filled / len(patient_fields) if filled > 0 else 0

            patient = PatientInfo(
                name=patient_data.get('name'),
                dob=patient_data.get('dob'),
                gender=patient_data.get('gender'),
                mrn=patient_data.get('mrn'),
                address=patient_data.get('address'),
                phone=patient_data.get('phone'),
                insurance=patient_data.get('insurance'),
                confidence=round(confidence, 2)
            )

            # Collect patient bounding box
            if collect_bbox:
                parsed = self._parse_bbox(patient_data)
                if parsed:
                    parsed.update({
                        "field": "Patient Info",
                        "value": patient_data.get('name', ''),
                        "confidence": round(confidence, 2),
                    })
                    bounding_boxes.append(parsed)

        # Parse test results
        test_results = []
        for item in llm_result.get('test_results', []):
            if isinstance(item, dict) and item.get('name'):
                value = item.get('value')
                if value is None or str(value).strip() in ('', 'null', 'None'):
                    continue

                test_fields = ['value', 'unit', 'reference_range']
                filled = sum(1 for f in test_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(test_fields))

                abnormal_flag = item.get('abnormal_flag')
                if abnormal_flag and str(abnormal_flag).lower() in ('null', 'none', ''):
                    abnormal_flag = None

                test_results.append(TestResult(
                    name=item.get('name', ''),
                    value=item.get('value'),
                    unit=item.get('unit'),
                    reference_range=item.get('reference_range'),
                    abnormal_flag=abnormal_flag,
                    category=item.get('category', 'lab'),
                    confidence=round(confidence, 2)
                ))

                # Collect bounding box
                if collect_bbox:
                    parsed = self._parse_bbox(item)
                    if parsed:
                        display_val = f"{item.get('value', '')}"
                        if item.get('unit'):
                            display_val += f" {item['unit']}"
                        parsed.update({
                            "field": item.get('name', ''),
                            "value": display_val,
                            "confidence": round(confidence, 2),
                        })
                        bounding_boxes.append(parsed)

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

        logger.info(f"Azure GPT-4o extracted {len(test_results)} test results")

        # Parse medications
        medications = []
        for item in llm_result.get('medications', []):
            if isinstance(item, dict) and item.get('name'):
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
                    confidence=round(confidence, 2)
                ))

                # Collect bounding box
                if collect_bbox:
                    parsed = self._parse_bbox(item)
                    if parsed:
                        parsed.update({
                            "field": item.get('name', ''),
                            "value": item.get('name', ''),
                            "confidence": round(confidence, 2),
                        })
                        bounding_boxes.append(parsed)

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
                finding_fields = ['category', 'severity', 'location']
                filled = sum(1 for f in finding_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(finding_fields))

                findings.append(ClinicalFinding(
                    finding=item.get('finding', ''),
                    category=item.get('category'),
                    severity=item.get('severity'),
                    location=item.get('location'),
                    confidence=round(confidence, 2)
                ))

                # Collect bounding box
                if collect_bbox:
                    parsed = self._parse_bbox(item)
                    if parsed:
                        parsed.update({
                            "field": item.get('finding', '')[:40],
                            "value": item.get('finding', ''),
                            "confidence": round(confidence, 2),
                        })
                        bounding_boxes.append(parsed)

        # Parse dates
        dates = []
        for item in llm_result.get('dates', []):
            if isinstance(item, dict) and item.get('date_value'):
                dates.append(DateInfo(
                    date_type=item.get('date_type', 'other'),
                    date_value=item.get('date_value', ''),
                    confidence=0.8
                ))

                # Collect bounding box
                if collect_bbox:
                    parsed = self._parse_bbox(item)
                    if parsed:
                        parsed.update({
                            "field": item.get('date_type', 'Date'),
                            "value": item.get('date_value', ''),
                            "confidence": 0.8,
                        })
                        bounding_boxes.append(parsed)

        # Parse providers
        providers = []
        for item in llm_result.get('providers', []):
            if isinstance(item, dict) and item.get('name'):
                provider_fields = ['role', 'specialty', 'npi']
                filled = sum(1 for f in provider_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(provider_fields))

                providers.append(ProviderInfo(
                    name=item.get('name', ''),
                    role=item.get('role'),
                    npi=item.get('npi'),
                    specialty=item.get('specialty'),
                    confidence=round(confidence, 2)
                ))

                # Collect bounding box
                if collect_bbox:
                    parsed = self._parse_bbox(item)
                    if parsed:
                        parsed.update({
                            "field": f"Provider: {item.get('name', '')}",
                            "value": item.get('name', ''),
                            "confidence": round(confidence, 2),
                        })
                        bounding_boxes.append(parsed)

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

                # Collect bounding box
                if collect_bbox:
                    parsed = self._parse_bbox(item)
                    if parsed:
                        parsed.update({
                            "field": item.get('name', ''),
                            "value": item.get('name', ''),
                            "confidence": round(confidence, 2),
                        })
                        bounding_boxes.append(parsed)

        # Parse procedures (dental, surgical, etc.)
        procedures = []
        for item in llm_result.get('procedures', []):
            if isinstance(item, dict) and item.get('procedure'):
                proc_fields = ['code', 'tooth', 'surface', 'date']
                filled = sum(1 for f in proc_fields if item.get(f))
                confidence = 0.5 + (0.5 * filled / len(proc_fields))

                procedures.append(ProcedureInfo(
                    name=item.get('procedure', ''),
                    date=item.get('date'),
                    provider=item.get('provider'),
                    findings=item.get('findings'),
                    confidence=round(confidence, 2)
                ))

                # Collect bounding box
                if collect_bbox:
                    parsed = self._parse_bbox(item)
                    if parsed:
                        parsed.update({
                            "field": item.get('procedure', ''),
                            "value": item.get('procedure', ''),
                            "confidence": round(confidence, 2),
                        })
                        bounding_boxes.append(parsed)

        # Capture financial/raw fields
        raw_fields = {}
        financial = llm_result.get('financial', {})
        if isinstance(financial, dict):
            if financial.get('invoice_number'):
                raw_fields['invoice_number'] = financial['invoice_number']
            if financial.get('total'):
                raw_fields['total_amount'] = financial['total']
            if financial.get('currency'):
                raw_fields['currency'] = financial['currency']
            # Capture line item amounts and their bounding boxes
            amounts = financial.get('amounts', [])
            if amounts and isinstance(amounts, list):
                raw_fields['line_items'] = amounts
                if collect_bbox:
                    for amt_item in amounts:
                        if isinstance(amt_item, dict):
                            parsed = self._parse_bbox(amt_item)
                            if parsed:
                                parsed.update({
                                    "field": amt_item.get('description', 'Line Item'),
                                    "value": amt_item.get('amount', ''),
                                    "confidence": 0.8,
                                })
                                bounding_boxes.append(parsed)

        # Capture document classification (used by pipeline for routing)
        doc_class = llm_result.get('document_classification', {})
        if isinstance(doc_class, dict):
            raw_fields['_classification_metadata'] = doc_class

        # Capture insurance details
        insurance_details = llm_result.get('insurance_details', {})
        if isinstance(insurance_details, dict):
            for key, value in insurance_details.items():
                if value and str(value).lower() not in ('null', 'none', ''):
                    raw_fields[f'insurance_{key}'] = value

        # Capture procedure-specific raw fields (tooth numbers, surfaces for dental)
        for item in llm_result.get('procedures', []):
            if isinstance(item, dict):
                if item.get('tooth'):
                    raw_fields.setdefault('teeth', []).append(item['tooth'])
                if item.get('code'):
                    raw_fields.setdefault('procedure_codes', []).append(item['code'])

        # Store bounding boxes in raw_fields for the pipeline to pass to frontend
        if collect_bbox and bounding_boxes:
            raw_fields['_bounding_boxes'] = bounding_boxes
            logger.info(f"Collected {len(bounding_boxes)} bounding boxes from GPT-4o")

        # Calculate confidence
        extraction = GenericMedicalExtraction(
            patient=patient,
            test_results=test_results,
            medications=medications,
            findings=findings,
            procedures=procedures,
            dates=dates,
            providers=providers,
            organizations=organizations,
            raw_fields=raw_fields,
            warnings=warnings
        )

        # Calculate real confidence based on completeness
        extraction.extraction_confidence = self._calculate_confidence(extraction)

        return extraction

    def _calculate_confidence(self, extraction: GenericMedicalExtraction) -> float:
        """Calculate confidence based on extraction completeness."""
        scores = []

        if extraction.patient:
            scores.append(extraction.patient.confidence)

        if extraction.test_results:
            avg = sum(t.confidence for t in extraction.test_results) / len(extraction.test_results)
            scores.append(avg)

        if extraction.medications:
            avg = sum(m.confidence for m in extraction.medications) / len(extraction.medications)
            scores.append(avg)

        if extraction.findings:
            avg = sum(f.confidence for f in extraction.findings) / len(extraction.findings)
            scores.append(avg)

        if extraction.procedures:
            avg = sum(p.confidence for p in extraction.procedures) / len(extraction.procedures)
            scores.append(avg)

        if extraction.dates:
            scores.append(0.8)

        if extraction.providers:
            avg = sum(p.confidence for p in extraction.providers) / len(extraction.providers)
            scores.append(avg)

        if extraction.organizations:
            avg = sum(o.confidence for o in extraction.organizations) / len(extraction.organizations)
            scores.append(avg)

        if not scores:
            return 0.2

        base_confidence = sum(scores) / len(scores)
        section_bonus = min(0.2, len(scores) * 0.03)

        return round(min(1.0, base_confidence + section_bonus), 3)


# Convenience function
async def extract_with_azure(
    file_path: Path,
    config: Dict[str, Any] = None
) -> GenericMedicalExtraction:
    """
    Convenience function for Azure OpenAI extraction.

    Args:
        file_path: Path to document file
        config: Optional configuration

    Returns:
        GenericMedicalExtraction with all extracted data
    """
    extractor = AzureOpenAIExtractor(config or {})
    return await extractor.extract_from_file(file_path)
