# ============================================================================
# src/medical_ingestion/processors/radiology/agents/diagnosis_extractor.py
# ============================================================================
"""
Diagnosis Extraction Agent for Radiology Reports

Extracts diagnoses and findings from radiology reports.
"""

from typing import List, Dict, Any
import logging
import re

from medical_ingestion.core.agent_base import Agent
from medical_ingestion.core.context.processing_context import ProcessingContext
from medical_ingestion.medgemma.client import create_client


class DiagnosisExtractor(Agent):
    """
    Extracts diagnoses and findings from radiology reports.

    Uses MedGemma to identify:
    - Primary diagnosis
    - Secondary findings
    - Incidental findings
    - Abnormalities
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.medgemma = create_client(config)
        self.logger = logging.getLogger(__name__)

    def get_name(self) -> str:
        return "DiagnosisExtractor"

    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract diagnoses from radiology report.

        Args:
            context: Processing context with report text

        Returns:
            Dict with extracted diagnoses
        """
        self.logger.info("Extracting diagnoses from radiology report")

        # Extract primary diagnosis
        primary = await self._extract_primary_diagnosis(context)

        # Extract secondary findings
        secondary = await self._extract_secondary_findings(context)

        # Extract incidental findings
        incidental = await self._extract_incidental_findings(context)

        # Check for abnormalities
        abnormalities = self._detect_abnormalities(context.raw_text)

        return {
            'primary_diagnosis': primary,
            'secondary_findings': secondary,
            'incidental_findings': incidental,
            'abnormalities': abnormalities
        }

    async def _extract_primary_diagnosis(
        self,
        context: ProcessingContext
    ) -> str:
        """
        Extract primary diagnosis using MedGemma.

        Args:
            context: Processing context

        Returns:
            Primary diagnosis text
        """
        prompt = f"""What is the primary diagnosis from this radiology report?

Report text:
{context.raw_text[:1500]}

Return only the primary diagnosis in 1 sentence."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1
            )
            return response['text'].strip()
        except Exception as e:
            self.logger.error(f"Primary diagnosis extraction failed: {e}")
            return "Unable to determine primary diagnosis"

    async def _extract_secondary_findings(
        self,
        context: ProcessingContext
    ) -> List[str]:
        """
        Extract secondary findings using MedGemma.

        Args:
            context: Processing context

        Returns:
            List of secondary findings
        """
        prompt = f"""List any secondary findings from this radiology report.

Report text:
{context.raw_text[:1500]}

Return as bullet points."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )

            # Parse bullet points
            findings = []
            for line in response['text'].split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    findings.append(line.lstrip('- •').strip())

            return findings
        except Exception as e:
            self.logger.error(f"Secondary findings extraction failed: {e}")
            return []

    async def _extract_incidental_findings(
        self,
        context: ProcessingContext
    ) -> List[str]:
        """
        Extract incidental findings using MedGemma.

        Args:
            context: Processing context

        Returns:
            List of incidental findings
        """
        prompt = f"""List any incidental findings from this radiology report.

Report text:
{context.raw_text[:1500]}

Return as bullet points or "None" if no incidental findings."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.1
            )

            if 'none' in response['text'].lower():
                return []

            # Parse bullet points
            findings = []
            for line in response['text'].split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    findings.append(line.lstrip('- •').strip())

            return findings
        except Exception as e:
            self.logger.error(f"Incidental findings extraction failed: {e}")
            return []

    def _detect_abnormalities(self, text: str) -> List[str]:
        """
        Detect abnormalities using keyword matching.

        Args:
            text: Report text

        Returns:
            List of detected abnormalities
        """
        abnormality_keywords = [
            'mass', 'lesion', 'nodule', 'opacity', 'consolidation',
            'effusion', 'pneumothorax', 'fracture', 'dislocation',
            'abnormal', 'suspicious', 'concerning', 'enlarged',
            'fluid', 'collection', 'hemorrhage', 'infarct'
        ]

        text_lower = text.lower()
        detected = []

        for keyword in abnormality_keywords:
            if keyword in text_lower:
                # Find context around keyword
                pattern = r'.{0,50}' + re.escape(keyword) + r'.{0,50}'
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    detected.append(f"{keyword}: {matches[0]}")

        return detected
