# ============================================================================
# src/medical_ingestion/processors/prescription/agents/medication_extractor.py
# ============================================================================
"""
Medication Extraction Agent

Extracts detailed medication information from prescriptions including:
- Drug names and RxNorm codes
- Dosage and strength
- Route of administration
- Frequency and duration
- Special instructions
"""

from typing import Dict, Any, Optional, List
import logging
import re

from medical_ingestion.core.agent_base import Agent
from medical_ingestion.core.context.processing_context import ProcessingContext
from medical_ingestion.medgemma.client import create_client


class MedicationExtractor(Agent):
    """
    Extracts medication information from prescription text.

    Supports:
    - Drug name normalization
    - RxNorm code mapping
    - Dosage parsing
    - Frequency parsing
    - Route identification
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.medgemma = create_client(config)
        self.logger = logging.getLogger(__name__)

        # Common administration routes
        self.routes = [
            'oral', 'po', 'by mouth',
            'iv', 'intravenous',
            'im', 'intramuscular',
            'sc', 'subcutaneous',
            'topical', 'transdermal',
            'inhaled', 'inhalation',
            'sublingual', 'sl',
            'rectal', 'pr',
            'ophthalmic', 'otic',
            'intranasal'
        ]

        # Common frequency patterns
        self.frequency_patterns = {
            'qd': 'once daily',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'q4h': 'every 4 hours',
            'q6h': 'every 6 hours',
            'q8h': 'every 8 hours',
            'q12h': 'every 12 hours',
            'qhs': 'at bedtime',
            'prn': 'as needed',
            'ac': 'before meals',
            'pc': 'after meals'
        }

    def get_name(self) -> str:
        return "MedicationExtractor"

    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract medication information from prescription.

        Args:
            context: Processing context with prescription text

        Returns:
            Dict with extracted medication data
        """
        self.logger.info("Extracting medication information from prescription")

        # Extract medication name
        drug_name = await self._extract_drug_name(context)

        # Get RxNorm code
        rxnorm_code = await self._get_rxnorm_code(drug_name)

        # Extract dosage/strength
        dosage = self._extract_dosage(context.raw_text)

        # Extract route
        route = self._extract_route(context.raw_text)

        # Extract frequency
        frequency = self._extract_frequency(context.raw_text)

        # Extract quantity and refills
        quantity = self._extract_quantity(context.raw_text)
        refills = self._extract_refills(context.raw_text)

        # Extract special instructions
        instructions = await self._extract_instructions(context)

        return {
            'drug_name': drug_name,
            'rxnorm_code': rxnorm_code,
            'dosage': dosage,
            'route': route,
            'frequency': frequency,
            'quantity': quantity,
            'refills': refills,
            'instructions': instructions
        }

    async def _extract_drug_name(
        self,
        context: ProcessingContext
    ) -> str:
        """
        Extract medication name using MedGemma.

        Args:
            context: Processing context

        Returns:
            Medication name
        """
        prompt = f"""What is the primary medication name in this prescription?

Prescription text:
{context.raw_text[:500]}

Return only the medication name (generic or brand name)."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1
            )
            return response['text'].strip()
        except Exception as e:
            self.logger.error(f"Drug name extraction failed: {e}")
            return "Unknown medication"

    async def _get_rxnorm_code(self, drug_name: str) -> Optional[str]:
        """
        Get RxNorm code for medication using MedGemma.

        Args:
            drug_name: Medication name

        Returns:
            RxNorm code or None
        """
        prompt = f"""What is the RxNorm code for: {drug_name}

Return only the RxNorm code number or "Unknown" if not found."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.1
            )

            code = response['text'].strip()
            if code.isdigit():
                return code

            return None
        except Exception as e:
            self.logger.error(f"RxNorm code lookup failed: {e}")
            return None

    def _extract_dosage(self, text: str) -> Dict[str, Any]:
        """
        Extract dosage/strength from prescription text.

        Args:
            text: Prescription text

        Returns:
            Dict with dosage information
        """
        # Common dosage patterns
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\b',
            r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)',
            r'(\d+(?:\.\d+)?)\s*%',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    return {
                        'value': float(match.group(1)),
                        'unit': match.group(2).lower(),
                        'text': match.group(0)
                    }
                elif len(match.groups()) == 3:
                    return {
                        'numerator': float(match.group(1)),
                        'denominator': float(match.group(2)),
                        'unit': match.group(3).lower(),
                        'text': match.group(0)
                    }
                else:
                    return {
                        'value': float(match.group(1)),
                        'unit': '%',
                        'text': match.group(0)
                    }

        return {}

    def _extract_route(self, text: str) -> str:
        """
        Extract route of administration.

        Args:
            text: Prescription text

        Returns:
            Route of administration
        """
        text_lower = text.lower()

        for route in self.routes:
            if route in text_lower:
                # Normalize common abbreviations
                if route in ['po', 'by mouth']:
                    return 'oral'
                elif route in ['iv']:
                    return 'intravenous'
                elif route in ['im']:
                    return 'intramuscular'
                elif route in ['sc']:
                    return 'subcutaneous'
                elif route in ['sl']:
                    return 'sublingual'
                elif route in ['pr']:
                    return 'rectal'
                else:
                    return route

        return 'oral'  # Default to oral

    def _extract_frequency(self, text: str) -> Dict[str, Any]:
        """
        Extract dosing frequency.

        Args:
            text: Prescription text

        Returns:
            Dict with frequency information
        """
        text_lower = text.lower()

        # Check for standard frequency abbreviations
        for abbrev, description in self.frequency_patterns.items():
            if abbrev in text_lower:
                return {
                    'code': abbrev,
                    'description': description,
                    'text': abbrev
                }

        # Check for numeric patterns
        numeric_patterns = [
            (r'(\d+)\s*times?\s*(?:per|a|daily)', 'times_per_day'),
            (r'every\s*(\d+)\s*hours?', 'every_n_hours'),
            (r'(\d+)\s*times?\s*(?:per|a)\s*week', 'times_per_week'),
        ]

        for pattern, freq_type in numeric_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = int(match.group(1))
                return {
                    'type': freq_type,
                    'value': value,
                    'text': match.group(0)
                }

        return {}

    def _extract_quantity(self, text: str) -> Optional[int]:
        """
        Extract quantity dispensed.

        Args:
            text: Prescription text

        Returns:
            Quantity or None
        """
        patterns = [
            r'(?:qty|quantity|disp|dispense)[:\s]*(\d+)',
            r'#(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _extract_refills(self, text: str) -> Optional[int]:
        """
        Extract number of refills.

        Args:
            text: Prescription text

        Returns:
            Number of refills or None
        """
        patterns = [
            r'(?:refills?|ref)[:\s]*(\d+)',
            r'(\d+)\s*refills?',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    async def _extract_instructions(
        self,
        context: ProcessingContext
    ) -> str:
        """
        Extract special instructions using MedGemma.

        Args:
            context: Processing context

        Returns:
            Special instructions text
        """
        prompt = f"""Extract any special patient instructions from this prescription.

Prescription text:
{context.raw_text[:500]}

Return only the special instructions or "None" if no special instructions."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.1
            )

            instructions = response['text'].strip()
            if 'none' in instructions.lower():
                return ''

            return instructions
        except Exception as e:
            self.logger.error(f"Instructions extraction failed: {e}")
            return ''
