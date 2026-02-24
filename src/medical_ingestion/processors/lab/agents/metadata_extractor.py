# ============================================================================
# src/medical_ingestion/processors/lab/agents/metadata_extractor.py
# ============================================================================
"""
Metadata Extraction Agent

Extracts document metadata including:
- Patient information (name, DOB, ID, demographics)
- Practitioner information (ordering physician, lab director)
- Organization information (performing lab, ordering facility)
- Specimen information (type, collection date, accession number)
- Report information (dates, status, order number)

This agent runs early in the pipeline to populate context with metadata
before lab value extraction begins.

Multi-page aware: Focuses on first page for header info, but scans
all pages for additional metadata that may appear elsewhere.
"""

import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, date
import logging

from medical_ingestion.core.agent_base import Agent

# from ...base_agent import Agent
from ....core.context import (
    ProcessingContext,
    PatientInfo,
    PractitionerInfo,
    OrganizationInfo,
    SpecimenInfo,
    ReportInfo,
    DocumentMetadata
)
from ....medgemma.client import create_client


# Common date patterns in lab reports
DATE_PATTERNS = [
    r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # MM/DD/YYYY or MM-DD-YYYY
    r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',    # YYYY-MM-DD
    r'([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})',  # Month DD, YYYY
]

# Common field labels in lab reports
PATIENT_LABELS = [
    r'patient\s*(?:name)?:?\s*',
    r'name:?\s*',
    r'pt\s*(?:name)?:?\s*',
]

DOB_LABELS = [
    r'(?:date\s+of\s+)?birth:?\s*',
    r'd\.?o\.?b\.?:?\s*',
    r'dob:?\s*',
    r'born:?\s*',
]

PATIENT_ID_LABELS = [
    r'patient\s*id:?\s*',
    r'mrn:?\s*',
    r'medical\s*record\s*(?:number|#)?:?\s*',
    r'account\s*(?:number|#)?:?\s*',
    r'acct\s*(?:number|#)?:?\s*',
]

PHYSICIAN_LABELS = [
    r'ordering\s*(?:physician|provider|doctor|md):?\s*',
    r'physician:?\s*',
    r'provider:?\s*',
    r'ordered\s*by:?\s*',
    r'requesting\s*(?:physician|provider):?\s*',
]

LAB_DIRECTOR_LABELS = [
    r'(?:lab(?:oratory)?|medical)\s*director:?\s*',
    r'pathologist:?\s*',
    r'reported\s*by:?\s*',
]

COLLECTION_DATE_LABELS = [
    r'collected:?\s*',
    r'collection\s*(?:date|time)?:?\s*',
    r'specimen\s*collected:?\s*',
    r'draw\s*date:?\s*',
]

RECEIVED_DATE_LABELS = [
    r'received:?\s*',
    r'received\s*(?:date|time)?:?\s*',
    r'specimen\s*received:?\s*',
]

REPORT_DATE_LABELS = [
    r'report(?:ed)?\s*(?:date)?:?\s*',
    r'resulted?:?\s*',
    r'final(?:ized)?:?\s*',
    r'printed:?\s*',
]

ACCESSION_LABELS = [
    r'accession\s*(?:number|#|no)?:?\s*',
    r'specimen\s*(?:id|number|#):?\s*',
    r'lab\s*(?:id|number|#):?\s*',
    r'order\s*(?:id|number|#):?\s*',
]


class MetadataExtractionAgent(Agent):
    """
    Extracts document metadata using a combination of:
    1. Regex patterns for common fields
    2. MedGemma for complex/ambiguous cases
    3. Template hints for vendor-specific formats
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.medgemma = create_client(config)
        self.logger = logging.getLogger(__name__)

    def get_name(self) -> str:
        return "MetadataExtractionAgent"

    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract all metadata from the document.

        Strategy:
        1. First try regex-based extraction (fast, reliable for standard formats)
        2. Use MedGemma for fields not found by regex
        3. Validate and cross-reference extracted data

        Returns:
            {
                "decision": "extracted",
                "confidence": float,
                "reasoning": str,
                "fields_extracted": int,
                "fields_missing": list
            }
        """
        self.logger.info(f"Extracting metadata from document: {context.document_id}")

        # Initialize document metadata
        context.ensure_metadata()

        # Get text - prefer page-aware text if available
        text = context.get_full_text_with_page_markers() if context.page_text else context.raw_text

        if not text:
            self.logger.warning("No text available for metadata extraction")
            return {
                "decision": "no_text",
                "confidence": 0.0,
                "reasoning": "No text available for metadata extraction",
                "fields_extracted": 0,
                "fields_missing": ["all"]
            }

        # Track what we find
        fields_extracted = 0
        fields_missing = []

        # ================================================================
        # PHASE 1: Regex-based extraction (fast)
        # ================================================================
        self.logger.info("Phase 1: Regex-based metadata extraction")

        # Focus on first page for header info (first 3000 chars typically)
        header_text = text[:3000] if len(text) > 3000 else text

        # Extract patient info
        patient = self._extract_patient_regex(header_text)
        if patient:
            context.set_patient_info(patient)
            fields_extracted += sum([
                1 for v in [patient.name, patient.patient_id, patient.date_of_birth, patient.age, patient.sex]
                if v is not None
            ])

        # Extract specimen info
        specimen = self._extract_specimen_regex(header_text)
        if specimen:
            context.set_specimen_info(specimen)
            fields_extracted += sum([
                1 for v in [specimen.accession_number, specimen.specimen_type, specimen.collection_date]
                if v is not None
            ])

        # Extract report info
        report = self._extract_report_regex(header_text, text)
        if report:
            context.set_report_info(report)
            fields_extracted += sum([
                1 for v in [report.collection_date, report.report_date, report.order_id]
                if v is not None
            ])

        # Extract practitioners
        ordering = self._extract_practitioner_regex(header_text, "ordering")
        if ordering:
            context.add_practitioner(ordering)
            fields_extracted += 1

        lab_director = self._extract_practitioner_regex(text, "lab_director")
        if lab_director:
            context.add_practitioner(lab_director)
            fields_extracted += 1

        # Extract organization (from template or text)
        org = self._extract_organization(header_text, context.template_id)
        if org:
            context.add_organization(org)
            fields_extracted += 1

        # ================================================================
        # PHASE 2: MedGemma for missing critical fields
        # ================================================================
        missing_critical = []
        if not context.document_metadata.patient or not context.document_metadata.patient.name:
            missing_critical.append("patient_name")
        if not context.document_metadata.patient or not context.document_metadata.patient.date_of_birth:
            missing_critical.append("patient_dob")
        if not context.document_metadata.ordering_provider:
            missing_critical.append("ordering_physician")
        if not context.document_metadata.report or not context.document_metadata.report.collection_date:
            missing_critical.append("collection_date")

        if missing_critical:
            self.logger.info(f"Phase 2: Using MedGemma for missing fields: {missing_critical}")
            medgemma_result = await self._extract_metadata_medgemma(header_text, missing_critical)

            if medgemma_result:
                self._apply_medgemma_results(context, medgemma_result)
                fields_extracted += len([v for v in medgemma_result.values() if v])

        # ================================================================
        # Calculate confidence and identify missing fields
        # ================================================================
        # Required fields for a complete extraction
        required = [
            ("patient_name", context.document_metadata.patient and context.document_metadata.patient.name),
            ("patient_dob", context.document_metadata.patient and context.document_metadata.patient.date_of_birth),
            ("collection_date", context.document_metadata.report and context.document_metadata.report.collection_date),
            ("ordering_provider", context.document_metadata.ordering_provider is not None),
            ("performing_lab", context.document_metadata.performing_lab is not None),
        ]

        for field_name, has_value in required:
            if not has_value:
                fields_missing.append(field_name)

        # Calculate confidence based on completeness
        completeness = (len(required) - len(fields_missing)) / len(required)
        confidence = min(0.95, 0.5 + (completeness * 0.45))

        # Update metadata stats
        context.document_metadata.fields_extracted = fields_extracted
        context.document_metadata.fields_missing = fields_missing
        context.document_metadata.overall_confidence = confidence

        self.logger.info(
            f"Metadata extraction complete: {fields_extracted} fields extracted, "
            f"{len(fields_missing)} missing, confidence: {confidence:.2f}"
        )

        # Flag for review if critical fields missing
        if "patient_name" in fields_missing or "collection_date" in fields_missing:
            context.requires_review = True
            context.review_reasons.append(f"Missing critical metadata: {fields_missing}")

        return {
            "decision": "extracted",
            "confidence": confidence,
            "reasoning": f"Extracted {fields_extracted} metadata fields, {len(fields_missing)} missing",
            "fields_extracted": fields_extracted,
            "fields_missing": fields_missing
        }

    # ========================================================================
    # REGEX-BASED EXTRACTION METHODS
    # ========================================================================

    def _extract_patient_regex(self, text: str) -> Optional[PatientInfo]:
        """Extract patient information using regex patterns."""
        patient = PatientInfo()

        # Extract patient name
        for pattern in PATIENT_LABELS:
            match = re.search(pattern + r'([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+)+)', text, re.IGNORECASE)
            if match:
                patient.name = match.group(1).strip()
                # Try to parse first/last
                parts = patient.name.split()
                if len(parts) >= 2:
                    patient.last_name = parts[-1]
                    patient.first_name = parts[0]
                    if len(parts) > 2:
                        patient.middle_name = ' '.join(parts[1:-1])
                break

        # Extract DOB
        for pattern in DOB_LABELS:
            for date_pattern in DATE_PATTERNS:
                match = re.search(pattern + date_pattern, text, re.IGNORECASE)
                if match:
                    try:
                        patient.date_of_birth = self._parse_date(match.groups()[-3:])
                        break
                    except (ValueError, IndexError):
                        continue
            if patient.date_of_birth:
                break

        # Calculate age from DOB
        if patient.date_of_birth:
            today = date.today()
            age = today.year - patient.date_of_birth.year
            if (today.month, today.day) < (patient.date_of_birth.month, patient.date_of_birth.day):
                age -= 1
            patient.age = age

        # Extract explicit age if DOB not found
        if not patient.age:
            age_match = re.search(r'age:?\s*(\d{1,3})\s*(?:y(?:ears?)?|yo)?', text, re.IGNORECASE)
            if age_match:
                patient.age = int(age_match.group(1))

        # Extract sex
        sex_match = re.search(r'(?:sex|gender):?\s*([MF](?:ale)?|Female?)', text, re.IGNORECASE)
        if sex_match:
            sex = sex_match.group(1).upper()[0]
            patient.sex = sex

        # Extract patient ID
        for pattern in PATIENT_ID_LABELS:
            match = re.search(pattern + r'([A-Z0-9]{4,20})', text, re.IGNORECASE)
            if match:
                patient.patient_id = match.group(1)
                break

        # Set confidence based on what we found
        found_count = sum(1 for v in [patient.name, patient.date_of_birth, patient.sex, patient.patient_id] if v)
        patient.confidence = min(0.95, 0.5 + (found_count * 0.15))

        return patient if found_count > 0 else None

    def _extract_specimen_regex(self, text: str) -> Optional[SpecimenInfo]:
        """Extract specimen information using regex patterns."""
        specimen = SpecimenInfo()

        # Extract accession number
        for pattern in ACCESSION_LABELS:
            match = re.search(pattern + r'([A-Z0-9\-]{6,20})', text, re.IGNORECASE)
            if match:
                specimen.accession_number = match.group(1)
                specimen.specimen_id = match.group(1)
                break

        # Extract specimen type
        type_match = re.search(
            r'(?:specimen|sample)\s*(?:type)?:?\s*(blood|serum|plasma|urine|whole blood|edta|lavender)',
            text, re.IGNORECASE
        )
        if type_match:
            specimen.specimen_type = type_match.group(1).title()

        # Extract collection date/time
        for pattern in COLLECTION_DATE_LABELS:
            for date_pattern in DATE_PATTERNS:
                match = re.search(pattern + date_pattern, text, re.IGNORECASE)
                if match:
                    try:
                        specimen.collection_date = self._parse_date(match.groups()[-3:])
                        break
                    except (ValueError, IndexError):
                        continue
            if specimen.collection_date:
                break

        # Extract fasting status
        fasting_match = re.search(r'fasting:?\s*(yes|no|y|n)', text, re.IGNORECASE)
        if fasting_match:
            specimen.fasting = fasting_match.group(1).lower() in ['yes', 'y']

        found_count = sum(1 for v in [specimen.accession_number, specimen.specimen_type, specimen.collection_date] if v)
        specimen.confidence = min(0.95, 0.5 + (found_count * 0.15))

        return specimen if found_count > 0 else None

    def _extract_report_regex(self, header_text: str, full_text: str) -> Optional[ReportInfo]:
        """Extract report information using regex patterns."""
        report = ReportInfo()

        # Collection date (from specimen section)
        for pattern in COLLECTION_DATE_LABELS:
            for date_pattern in DATE_PATTERNS:
                match = re.search(pattern + date_pattern, header_text, re.IGNORECASE)
                if match:
                    try:
                        report.collection_date = self._parse_date(match.groups()[-3:])
                        break
                    except (ValueError, IndexError):
                        continue
            if report.collection_date:
                break

        # Received date
        for pattern in RECEIVED_DATE_LABELS:
            for date_pattern in DATE_PATTERNS:
                match = re.search(pattern + date_pattern, header_text, re.IGNORECASE)
                if match:
                    try:
                        report.received_date = self._parse_date(match.groups()[-3:])
                        break
                    except (ValueError, IndexError):
                        continue
            if report.received_date:
                break

        # Report date (often at end of document)
        for pattern in REPORT_DATE_LABELS:
            for date_pattern in DATE_PATTERNS:
                # Search end of document for report date
                search_text = full_text[-2000:] if len(full_text) > 2000 else full_text
                match = re.search(pattern + date_pattern, search_text, re.IGNORECASE)
                if match:
                    try:
                        report.report_date = self._parse_date(match.groups()[-3:])
                        break
                    except (ValueError, IndexError):
                        continue
            if report.report_date:
                break

        # Order number
        order_match = re.search(r'order\s*(?:number|#|no)?:?\s*([A-Z0-9\-]{6,20})', header_text, re.IGNORECASE)
        if order_match:
            report.order_id = order_match.group(1)

        # Status indicators
        if re.search(r'\b(?:preliminary|pending)\b', header_text, re.IGNORECASE):
            report.status = "preliminary"
            report.is_preliminary = True
        elif re.search(r'\b(?:corrected|amended)\b', header_text, re.IGNORECASE):
            report.status = "corrected"
            report.is_corrected = True
        else:
            report.status = "final"

        # Page count
        page_match = re.search(r'page\s*\d+\s*of\s*(\d+)', full_text, re.IGNORECASE)
        if page_match:
            report.page_count = int(page_match.group(1))

        found_count = sum(1 for v in [report.collection_date, report.report_date, report.order_id] if v)
        report.confidence = min(0.95, 0.5 + (found_count * 0.15))

        return report if found_count > 0 else None

    def _extract_practitioner_regex(self, text: str, role: str) -> Optional[PractitionerInfo]:
        """Extract practitioner information."""
        practitioner = PractitionerInfo(role=role)

        labels = PHYSICIAN_LABELS if role == "ordering" else LAB_DIRECTOR_LABELS

        for pattern in labels:
            # Match name with possible credentials
            match = re.search(
                pattern + r'([A-Z][a-zA-Z\'-]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-zA-Z\'-]+)(?:,?\s*(M\.?D\.?|D\.?O\.?|Ph\.?D\.?))?',
                text, re.IGNORECASE
            )
            if match:
                practitioner.name = match.group(1).strip()
                if match.group(2):
                    practitioner.credentials = match.group(2).replace('.', '')
                break

        # Extract NPI if present
        npi_match = re.search(r'npi:?\s*(\d{10})', text, re.IGNORECASE)
        if npi_match:
            practitioner.npi = npi_match.group(1)

        if practitioner.name:
            practitioner.confidence = 0.85
            return practitioner
        return None

    def _extract_organization(self, text: str, template_id: Optional[str]) -> Optional[OrganizationInfo]:
        """Extract organization information from text or template."""
        org = OrganizationInfo(role="performing")

        # Try to infer from template ID first
        if template_id:
            if 'labcorp' in template_id.lower():
                org.name = "LabCorp"
                org.display_name = "Laboratory Corporation of America"
            elif 'quest' in template_id.lower():
                org.name = "Quest Diagnostics"
                org.display_name = "Quest Diagnostics Incorporated"

        # Look for lab name in text
        if not org.name:
            lab_patterns = [
                r'(labcorp|laboratory\s+corporation)',
                r'(quest\s+diagnostics)',
                r'(bioreference|bio-?reference)',
                r'(sonic\s+healthcare)',
            ]
            for pattern in lab_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    org.name = match.group(1).strip()
                    break

        # Extract CLIA number
        clia_match = re.search(r'clia(?:\s*#?)?:?\s*(\d{2}D?\d{7})', text, re.IGNORECASE)
        if clia_match:
            org.clia_number = clia_match.group(1)

        # Extract phone
        phone_match = re.search(r'(?:phone|tel|call):?\s*(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', text, re.IGNORECASE)
        if phone_match:
            org.phone = phone_match.group(1)

        if org.name or org.clia_number:
            org.confidence = 0.9
            return org
        return None

    def _parse_date(self, groups: tuple) -> Optional[date]:
        """Parse date from regex groups."""
        try:
            # Handle different formats
            if len(groups) >= 3:
                part1, part2, part3 = groups[-3:]

                # Check if month is text
                months = {
                    'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
                    'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
                    'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                    'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                    'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
                    'dec': 12, 'december': 12
                }

                if part1.lower() in months:
                    # Month DD, YYYY
                    month = months[part1.lower()]
                    day = int(part2)
                    year = int(part3)
                elif len(part1) == 4:
                    # YYYY-MM-DD
                    year = int(part1)
                    month = int(part2)
                    day = int(part3)
                else:
                    # MM/DD/YYYY
                    month = int(part1)
                    day = int(part2)
                    year = int(part3)

                # Handle 2-digit year
                if year < 100:
                    year = 2000 + year if year < 50 else 1900 + year

                return date(year, month, day)
        except (ValueError, TypeError):
            pass
        return None

    # ========================================================================
    # MEDGEMMA-BASED EXTRACTION
    # ========================================================================

    async def _extract_metadata_medgemma(self, text: str, missing_fields: List[str]) -> Dict[str, Any]:
        """Use MedGemma to extract missing metadata fields."""
        # Limit text for prompt
        text_sample = text[:4000]

        prompt = f"""Extract the following information from this medical lab report.
Return ONLY a JSON object with the requested fields.

DOCUMENT TEXT:
{text_sample}

FIELDS TO EXTRACT:
{json.dumps(missing_fields)}

Return this exact JSON format (use null for fields not found):
{{
  "patient_name": "First Last" or null,
  "patient_dob": "YYYY-MM-DD" or null,
  "patient_age": number or null,
  "patient_sex": "M" or "F" or null,
  "ordering_physician": "Dr. Name, MD" or null,
  "collection_date": "YYYY-MM-DD" or null,
  "report_date": "YYYY-MM-DD" or null,
  "accession_number": "string" or null,
  "specimen_type": "Blood", "Serum", etc. or null,
  "lab_name": "LabCorp", "Quest", etc. or null
}}

JSON:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
                json_mode=True
            )

            result_text = response.get('text', '').strip()

            if result_text:
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if match:
                        return json.loads(match.group())
        except Exception as e:
            self.logger.warning(f"MedGemma metadata extraction failed: {e}")

        return {}

    def _apply_medgemma_results(self, context: ProcessingContext, results: Dict[str, Any]):
        """Apply MedGemma extraction results to context."""
        metadata = context.ensure_metadata()

        # Patient info
        if results.get("patient_name") and not (metadata.patient and metadata.patient.name):
            if not metadata.patient:
                metadata.patient = PatientInfo()
            metadata.patient.name = results["patient_name"]

        if results.get("patient_dob") and not (metadata.patient and metadata.patient.date_of_birth):
            if not metadata.patient:
                metadata.patient = PatientInfo()
            try:
                metadata.patient.date_of_birth = datetime.strptime(
                    results["patient_dob"], "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                pass

        if results.get("patient_age") and not (metadata.patient and metadata.patient.age):
            if not metadata.patient:
                metadata.patient = PatientInfo()
            metadata.patient.age = int(results["patient_age"])

        if results.get("patient_sex") and not (metadata.patient and metadata.patient.sex):
            if not metadata.patient:
                metadata.patient = PatientInfo()
            metadata.patient.sex = results["patient_sex"]

        # Ordering physician
        if results.get("ordering_physician") and not metadata.ordering_provider:
            metadata.ordering_provider = PractitionerInfo(
                role="ordering",
                name=results["ordering_physician"],
                confidence=0.75
            )

        # Dates
        if results.get("collection_date") and not (metadata.report and metadata.report.collection_date):
            if not metadata.report:
                metadata.report = ReportInfo()
            try:
                metadata.report.collection_date = datetime.strptime(
                    results["collection_date"], "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                pass

        if results.get("report_date") and not (metadata.report and metadata.report.report_date):
            if not metadata.report:
                metadata.report = ReportInfo()
            try:
                metadata.report.report_date = datetime.strptime(
                    results["report_date"], "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                pass

        # Specimen
        if results.get("accession_number") and not (metadata.specimen and metadata.specimen.accession_number):
            if not metadata.specimen:
                metadata.specimen = SpecimenInfo()
            metadata.specimen.accession_number = results["accession_number"]

        if results.get("specimen_type") and not (metadata.specimen and metadata.specimen.specimen_type):
            if not metadata.specimen:
                metadata.specimen = SpecimenInfo()
            metadata.specimen.specimen_type = results["specimen_type"]

        # Lab
        if results.get("lab_name") and not metadata.performing_lab:
            metadata.performing_lab = OrganizationInfo(
                role="performing",
                name=results["lab_name"],
                confidence=0.75
            )
