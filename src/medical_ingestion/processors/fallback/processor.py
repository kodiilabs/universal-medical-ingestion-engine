# src/medical_ingestion/processors/fallback/processor.py
"""
Fallback Processor - Universal Document Handler

Handles documents that:
1. Cannot be classified with high confidence
2. Are classified but have no matching templates
3. Are of unknown/unsupported document types

Strategy:
1. Use MedGemma to extract what it can (best-effort)
2. Generate draft template for admin review
3. Flag for human review
4. Return partial results with low confidence

This ensures every document gets some processing, even unknown formats.
"""

from typing import Dict, Any, List

from json_repair import repair_json

from ..base_processor import BaseProcessor
from ...core.context.processing_context import ProcessingContext
from ...extractors.universal_text_extractor import UniversalTextExtractor
from ...medgemma.client import create_client

# Optional: Template generator for auto-creating templates from unknown formats
try:
    from ..template_generator import TemplateGenerator
    HAS_TEMPLATE_GENERATOR = True
except ImportError:
    HAS_TEMPLATE_GENERATOR = False
    TemplateGenerator = None

# Confidence penalty when json_repair is used
JSON_REPAIR_CONFIDENCE_PENALTY = 0.10


class FallbackProcessor(BaseProcessor):
    """
    Universal fallback processor for unknown document types.

    Handles any document that other processors cannot process,
    using MedGemma for best-effort extraction and auto-template generation.
    """

    def get_name(self) -> str:
        return "FallbackProcessor"

    def _get_agents(self) -> List:
        """
        Fallback processor uses direct MedGemma calls rather than agents.
        """
        return []

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Process unknown document with best-effort extraction.

        Returns:
            {
                "success": bool,
                "extraction_method": "fallback",
                "extracted_fields": int,
                "confidence": float,
                "requires_review": True,
                "auto_template": {...}
            }
        """
        self.logger.info(
            f"Fallback processing document: {context.document_id} "
            f"(type: {context.document_type or 'unknown'})"
        )

        try:
            # Step 1: Extract text from document using UniversalTextExtractor
            # This handles PDFs, images, scanned docs with proper OCR routing
            if not context.raw_text:
                text_extractor = UniversalTextExtractor(self.config)
                extraction_result = await text_extractor.extract(context.document_path)
                text = extraction_result.full_text
                context.raw_text = text
                context.total_pages = extraction_result.page_count
            else:
                text = context.raw_text

            # Step 2: Extract ALL key-value pairs directly from raw text
            # This catches everything visible, regardless of MedGemma's schema
            raw_extraction_count = self._extract_all_from_raw_text(context, text)
            self.logger.info(f"Raw text extraction: {raw_extraction_count} fields")

            # Step 3: Use MedGemma to add semantic understanding and structured extraction
            extraction_result = await self._extract_with_medgemma(context, text)

            # Step 4: Generate draft template for future use
            template_result = await self._generate_draft_template(context, text)

            # Step 4: Flag for human review
            context.requires_review = True
            context.review_reasons.append(
                f"Fallback processing used - document type '{context.document_type or 'unknown'}' "
                f"requires template creation"
            )
            context.add_warning(
                f"No template available for document type: {context.document_type or 'unknown'}"
            )

            total_fields = raw_extraction_count + extraction_result.get('field_count', 0)
            return {
                "success": True,
                "extraction_method": "fallback",
                "extracted_fields": total_fields,
                "raw_text_fields": raw_extraction_count,
                "medgemma_fields": extraction_result.get('field_count', 0),
                "confidence": extraction_result.get('confidence', 0.5),
                "requires_review": True,
                "auto_template": template_result,
                "notes": [
                    f"Raw text parsing extracted {raw_extraction_count} fields",
                    f"MedGemma semantic extraction added {extraction_result.get('field_count', 0)} structured fields",
                    "Manual verification recommended"
                ]
            }

        except Exception as e:
            self.logger.error(f"Fallback processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "requires_review": True
            }

    async def _extract_with_medgemma(
        self,
        context: ProcessingContext,
        text: str
    ) -> Dict[str, Any]:
        """
        Use MedGemma for generic medical document extraction.
        """
        from ...core.context import ExtractedValue

        client = create_client(self.config)

        # Truncate text if too long
        text_sample = text[:4000]

        # Determine document type hint
        doc_type = context.document_type or 'medical document'

        prompt = f"""Analyze this {doc_type} and extract ALL structured data. Be extremely thorough.

Document text:
{text_sample}

Extract EVERY piece of data as JSON. Look carefully for:

PEOPLE & ORGANIZATIONS:
- Patient/participant/claimant name and ID
- Provider/practitioner name, credentials, license/registration numbers
- Facility/clinic/company name, address, phone, email
- Any other names mentioned

IDENTIFIERS:
- Claim numbers, invoice numbers, reference numbers
- Policy numbers, member IDs, account numbers
- Registration numbers, license numbers

DATES:
- Service/session date and time
- Submission/report date
- Payment date
- Any other dates

AMOUNTS:
- Service fees, session costs
- Subtotals, totals, payments
- Any monetary values with currency

SERVICES/LINE ITEMS:
- Service description (e.g., "Counselling Session 60 min")
- Service type/category (e.g., "Paramedical", "Psychotherapy")
- Quantity, duration

PAYMENT DETAILS:
- Payment method (cash, credit card, etc.)
- Card type and last 4 digits if shown
- Payment status

OTHER:
- Coverage information
- Benefit type
- Any labeled field: value pairs

Return JSON:
{{
    "document_type": "specific type (e.g., paramedical_claim, counselling_receipt)",
    "patient": {{
        "name": "full name",
        "id": "participant/member ID"
    }},
    "provider": {{
        "name": "practitioner name",
        "credentials": "license/registration",
        "organization": "clinic/facility name",
        "address": "full address",
        "phone": "phone number",
        "email": "email address"
    }},
    "claim": {{
        "type": "claim type",
        "benefit": "benefit category",
        "submission_date": "date",
        "status": "status if shown"
    }},
    "service": {{
        "description": "what service was provided",
        "date": "service date",
        "duration": "duration if applicable"
    }},
    "amounts": {{
        "service_fee": "amount",
        "subtotal": "amount",
        "total": "amount",
        "payment": "amount paid"
    }},
    "payment": {{
        "method": "cash/credit card/etc",
        "card_type": "VISA/MASTERCARD/etc",
        "card_last4": "last 4 digits",
        "date": "payment date",
        "status": "authorized/completed/etc"
    }},
    "identifiers": {{"id_type": "value"}},
    "dates": {{"date_label": "date_value"}},
    "raw_fields": {{"any_other_field": "value"}}
}}

Extract EVERYTHING. Do not skip ANY data visible in the document.
JSON:"""

        try:
            # Use json_mode=True to constrain output to valid JSON
            response = await client.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1,
                json_mode=True
            )

            # Parse JSON response
            result, json_repaired = self._parse_json_response(response.get('text', ''))

            if not result:
                return {'field_count': 0, 'confidence': 0.3}

            # Determine confidence based on whether json_repair was used
            base_confidence = 0.6
            if json_repaired:
                base_confidence -= JSON_REPAIR_CONFIDENCE_PENALTY
                self.logger.warning(
                    f"JSON repair was used - confidence reduced to {base_confidence:.2f}. "
                    "Some data may have been silently corrected or lost."
                )
                context.warnings.append(
                    "MedGemma output required JSON repair - verify extracted values"
                )
                context.requires_review = True
                context.review_reasons.append(
                    "JSON repair was used during extraction - potential data loss"
                )

            # Store extracted data in context
            field_count = 0
            extraction_method = "fallback_medgemma" + ("_repaired" if json_repaired else "")

            def add_field(name: str, value, unit: str = '', conf: float = base_confidence):
                """Helper to add extracted value."""
                nonlocal field_count
                if value is not None and str(value).strip():
                    extracted = ExtractedValue(
                        field_name=name.lower().replace(' ', '_'),
                        value=str(value).strip(),
                        unit=unit,
                        confidence=conf,
                        extraction_method=extraction_method
                    )
                    context.add_extracted_value(extracted)
                    field_count += 1

            def process_dict(data: dict, prefix: str):
                """Process a nested dict, flattening keys with prefix."""
                if not isinstance(data, dict):
                    return
                for key, value in data.items():
                    if value is not None and str(value).strip():
                        if isinstance(value, dict):
                            process_dict(value, f"{prefix}_{key}")
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                if isinstance(item, dict):
                                    process_dict(item, f"{prefix}_{key}_{i}")
                                else:
                                    add_field(f"{prefix}_{key}_{i}", item)
                        else:
                            add_field(f"{prefix}_{key}", value)

            # Process structured sections
            for section in ['patient', 'provider', 'claim', 'service', 'payment']:
                if section in result and result[section]:
                    process_dict(result[section], section)
                    # Also store in sections for easy access
                    context.sections[section] = result[section]

            # Process amounts (can be dict or list)
            amounts_data = result.get('amounts', {})
            if isinstance(amounts_data, dict):
                for key, value in amounts_data.items():
                    add_field(f"amount_{key}", value, unit='$')
            elif isinstance(amounts_data, list):
                for i, amount in enumerate(amounts_data):
                    if isinstance(amount, dict) and amount.get('value'):
                        desc = amount.get('description', f'item_{i}')
                        add_field(f"amount_{desc[:30]}", amount['value'], amount.get('currency', '$'))
                    else:
                        add_field(f"amount_{i}", amount)

            # Process identifiers
            for id_type, value in result.get('identifiers', {}).items():
                add_field(f"id_{id_type}", value)

            # Process dates
            for date_type, value in result.get('dates', {}).items():
                add_field(f"date_{date_type}", value)

            # Process results array (test results, measurements)
            for item in result.get('results', []):
                if item.get('field') and item.get('value') is not None:
                    add_field(item['field'], item['value'], item.get('unit', ''))

            # Store raw fields (slightly lower confidence)
            raw_confidence = base_confidence - 0.1
            for field_name, value in result.get('raw_fields', {}).items():
                add_field(field_name, value, conf=raw_confidence)

            # Store entities, line_items, findings in sections
            for key in ['entities', 'line_items', 'findings']:
                if result.get(key):
                    context.sections[key] = result[key]

            # Store detected document type
            if result.get('document_type'):
                context.sections['detected_type'] = result['document_type']

            return {
                'field_count': field_count,
                'confidence': base_confidence if field_count > 0 else 0.3,
                'detected_type': result.get('document_type'),
                'json_repaired': json_repaired
            }

        except Exception as e:
            self.logger.error(f"MedGemma extraction failed: {e}")
            return {'field_count': 0, 'confidence': 0.2, 'error': str(e)}

    def _extract_all_from_raw_text(
        self,
        context: ProcessingContext,
        text: str
    ) -> int:
        """
        Extract ALL data directly from raw text.

        This extracts EVERYTHING visible in the document:
        - Key-value pairs (Label: Value, Label / French: Value)
        - Standalone data (emails, phones, amounts, dates)
        - Addresses
        - Transaction IDs
        - Any identifiable structured data

        Returns:
            Number of fields extracted
        """
        import re
        from ...core.context import ExtractedValue

        field_count = 0
        extraction_method = "raw_text_parsing"
        confidence = 0.85

        extracted_values = set()

        def add_raw_field(name: str, value: str, conf: float = confidence):
            nonlocal field_count
            name = name.strip().lower().replace(' ', '_').replace('/', '_')
            name = re.sub(r'[^a-z0-9_]', '', name)[:50]  # Limit name length
            value = value.strip()

            if not name or not value or len(value) < 2:
                return

            # Skip duplicates
            key = f"{name}:{value[:50]}"
            if key in extracted_values:
                return
            extracted_values.add(key)

            extracted = ExtractedValue(
                field_name=f"raw_{name}",
                value=value,
                unit='',
                confidence=conf,
                extraction_method=extraction_method
            )
            context.add_extracted_value(extracted)
            field_count += 1

        # ===== PATTERN MATCHING =====

        # Email addresses
        for match in re.findall(r'[\w.-]+@[\w.-]+\.\w+', text):
            add_raw_field('email', match)

        # Phone numbers (various formats)
        for match in re.findall(r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b', text):
            add_raw_field('phone', match)

        # Amounts with $ symbol
        for match in re.findall(r'\$[\d,]+\.?\d*', text):
            add_raw_field('amount', match)

        # ISO dates (2025-02-26)
        for match in re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text):
            add_raw_field('date_iso', match)

        # Text dates (February 26, 2025)
        for match in re.findall(
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            text, re.IGNORECASE
        ):
            add_raw_field('date_text', match)

        # Invoice/reference numbers (patterns like #CVC14063-P01, #11248142)
        for match in re.findall(r'#[A-Za-z0-9-]+', text):
            add_raw_field('reference_number', match)

        # Transaction IDs (patterns like ch_3QwsBjJnuK6...)
        for match in re.findall(r'\b(?:ch|pi|in|cus)_[A-Za-z0-9]{10,}\b', text):
            add_raw_field('transaction_id', match)

        # UUIDs
        for match in re.findall(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b', text, re.IGNORECASE):
            add_raw_field('uuid', match)

        # Credit card info (last 4 digits pattern)
        for match in re.findall(r'(?:VISA|MASTERCARD|AMEX|DISCOVER)\s+ending\s+in\s+\d{4}', text, re.IGNORECASE):
            add_raw_field('card_info', match)

        # Postal codes (Canadian: V9N 8R9, US: 12345 or 12345-6789)
        for match in re.findall(r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b', text, re.IGNORECASE):
            add_raw_field('postal_code', match)

        # Registration/license numbers (patterns with # followed by numbers)
        for match in re.findall(r'(?:RT|GST|HST|PST|QST)\d+', text, re.IGNORECASE):
            add_raw_field('tax_registration', match)

        # ===== KEY-VALUE EXTRACTION =====

        # Split by common delimiters and extract pairs
        # Pattern: "Label / French Label : Value" or just "Label: Value"
        kv_pattern = re.compile(
            r'([A-Za-z][A-Za-zéèêëàâäùûüôöîïç\s]{2,40})'  # Label (2-40 chars)
            r'(?:\s*/\s*[A-Za-zéèêëàâäùûüôöîïç\s]+)?'  # Optional French label
            r'\s*:\s*'  # Colon separator
            r'([^\n]{2,150})',  # Value (2-150 chars)
            re.UNICODE
        )

        for match in kv_pattern.finditer(text):
            label, value = match.groups()
            # Clean the label - take only English part if bilingual
            label = label.strip()
            if '/' in label:
                label = label.split('/')[0].strip()
            # Clean value - stop at next label
            value = re.split(r'\s+[A-Z][a-z]+\s*[/:]', value)[0].strip()
            if value and len(value) >= 2:
                add_raw_field(label, value)

        # ===== ADDRESS EXTRACTION =====

        # Canadian addresses (Number Street, City, Province, Postal)
        addr_pattern = re.compile(
            r'(\d+(?:\s*-?\s*\d*[A-Z]?)?\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Crescent|Cres|Boulevard|Blvd|Way|Lane|Ln)[,\s]+[A-Za-z\s]+,\s*[A-Z]{2},?\s*[A-Z]\d[A-Z]\s*\d[A-Z]\d)',
            re.IGNORECASE
        )
        for match in addr_pattern.findall(text):
            add_raw_field('address', match)

        # Also try to extract just street addresses
        street_pattern = re.compile(r'\b(\d+\s+[A-Za-z\s]+(?:RD|ROAD|ST|STREET|AVE|AVENUE|DR|DRIVE|CRES|CRESCENT))\b', re.IGNORECASE)
        for match in street_pattern.findall(text):
            add_raw_field('street_address', match)

        # City, Province, Postal
        city_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z]{2})\s+([A-Z]\d[A-Z]\s*\d[A-Z]\d)\b')
        for match in city_pattern.findall(text):
            city, province, postal = match
            add_raw_field('city', city)
            add_raw_field('province', province)
            add_raw_field('postal_code_extracted', postal)

        # Store the full raw text
        context.sections['raw_text'] = text

        return field_count

    async def _generate_draft_template(
        self,
        context: ProcessingContext,
        text: str
    ) -> Dict[str, Any]:
        """
        Generate draft template for unknown document format.
        """
        if not HAS_TEMPLATE_GENERATOR:
            self.logger.debug("TemplateGenerator not available - skipping draft template generation")
            return {'error': 'TemplateGenerator not available', 'needs_review': True}

        try:
            generator = TemplateGenerator(self.config)

            # Use classified type or 'unknown'
            doc_type = context.document_type or 'unknown'

            # Get visual info if available
            visual_info = context.sections.get('visual_identification')

            result = await generator.generate_template(
                pdf_path=context.document_path,
                document_type=doc_type,
                extracted_text=text,
                visual_info=visual_info
            )

            self.logger.info(
                f"Generated draft template: {result.get('template', {}).get('id')} "
                f"(confidence: {result.get('confidence', 0):.2f})"
            )

            # Store in context
            context.sections['auto_template'] = {
                'draft_path': str(result.get('draft_path', '')),
                'template_id': result.get('template', {}).get('id'),
                'confidence': result.get('confidence', 0),
                'needs_review': True
            }

            return {
                'template_id': result.get('template', {}).get('id'),
                'draft_path': str(result.get('draft_path', '')),
                'confidence': result.get('confidence', 0),
                'needs_review': True,
                'validation_notes': result.get('validation_notes', [])
            }

        except Exception as e:
            self.logger.error(f"Template generation failed: {e}")
            return {
                'error': str(e),
                'needs_review': True
            }

    def _parse_json_response(self, text: str) -> tuple[Dict[str, Any], bool]:
        """
        Extract JSON from model response with repair fallback.

        Returns:
            Tuple of (parsed_dict, json_was_repaired)
        """
        import json

        # Try 1: Direct parse
        try:
            return json.loads(text), False
        except json.JSONDecodeError:
            pass

        # Try 2: Use json_repair
        try:
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                self.logger.warning(
                    f"json_repair fixed response - potential data loss. "
                    f"Original (first 200 chars): {text[:200]}"
                )
                return repaired, True
        except Exception:
            pass

        # Try 3: Extract JSON block by braces
        try:
            start = text.find('{')
            if start == -1:
                return {}, False

            depth = 0
            end = start
            for i, char in enumerate(text[start:], start=start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break

            json_str = text[start:end + 1]

            try:
                return json.loads(json_str), False
            except json.JSONDecodeError:
                # Try repair on extracted block
                try:
                    repaired = repair_json(json_str, return_objects=True)
                    if isinstance(repaired, dict):
                        self.logger.warning(
                            f"json_repair fixed extracted JSON block - potential data loss. "
                            f"Original (first 200 chars): {json_str[:200]}"
                        )
                        return repaired, True
                except Exception:
                    pass

        except Exception as e:
            self.logger.warning(f"Failed to parse JSON: {e}")

        return {}, False
