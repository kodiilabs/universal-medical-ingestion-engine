# ============================================================================
# src/medical_ingestion/processors/radiology/agents/staging_agent.py
# ============================================================================
"""
Staging Agent for Radiology Reports

Extracts cancer staging information from radiology/oncology reports.
"""

from typing import Dict, Any, Optional
import logging
import re

from medical_ingestion.core.agent_base import Agent
from medical_ingestion.core.context.processing_context import ProcessingContext
from medical_ingestion.medgemma.client import create_client


class StagingAgent(Agent):
    """
    Extracts cancer staging information from radiology reports.

    Supports:
    - TNM staging (Tumor, Node, Metastasis)
    - RECIST criteria (tumor response)
    - Lesion measurements
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.medgemma = create_client(config)
        self.logger = logging.getLogger(__name__)

    def get_name(self) -> str:
        return "StagingAgent"

    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract staging information from radiology report.

        Args:
            context: Processing context with report text

        Returns:
            Dict with staging information
        """
        self.logger.info("Extracting staging information from radiology report")

        # Check if report contains cancer/tumor content
        if not self._is_oncology_report(context.raw_text):
            return {
                'is_oncology': False,
                'tnm_stage': None,
                'measurements': [],
                'response_assessment': None
            }

        # Extract TNM staging
        tnm = self._extract_tnm_staging(context.raw_text)

        # Extract lesion measurements
        measurements = self._extract_measurements(context.raw_text)

        # Extract response assessment
        response = await self._extract_response_assessment(context)

        return {
            'is_oncology': True,
            'tnm_stage': tnm,
            'measurements': measurements,
            'response_assessment': response
        }

    def _is_oncology_report(self, text: str) -> bool:
        """
        Check if report is oncology-related.

        Args:
            text: Report text

        Returns:
            True if oncology report
        """
        oncology_keywords = [
            'tumor', 'cancer', 'carcinoma', 'adenocarcinoma',
            'metastasis', 'metastatic', 'malignancy', 'malignant',
            'mass', 'lesion', 'nodule', 'staging', 'tnm'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in oncology_keywords)

    def _extract_tnm_staging(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract TNM staging from report text.

        Args:
            text: Report text

        Returns:
            Dict with T, N, M components or None
        """
        # Common TNM patterns
        tnm_pattern = r'(T[0-4][a-c]?)\s*(N[0-3][a-c]?)\s*(M[0-1][a-c]?)'

        match = re.search(tnm_pattern, text, re.IGNORECASE)
        if match:
            return {
                'T': match.group(1).upper(),
                'N': match.group(2).upper(),
                'M': match.group(3).upper(),
                'full': f"{match.group(1)}{match.group(2)}{match.group(3)}".upper()
            }

        # Try individual components
        t_match = re.search(r'T[0-4][a-c]?', text, re.IGNORECASE)
        n_match = re.search(r'N[0-3][a-c]?', text, re.IGNORECASE)
        m_match = re.search(r'M[0-1][a-c]?', text, re.IGNORECASE)

        if t_match or n_match or m_match:
            return {
                'T': t_match.group(0).upper() if t_match else None,
                'N': n_match.group(0).upper() if n_match else None,
                'M': m_match.group(0).upper() if m_match else None,
                'full': None
            }

        return None

    def _extract_measurements(self, text: str) -> list:
        """
        Extract lesion/tumor measurements from report.

        Args:
            text: Report text

        Returns:
            List of measurement dicts
        """
        measurements = []

        # Pattern for measurements: "X.X x Y.Y cm" or "X.X cm"
        patterns = [
            r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*cm',  # 3D
            r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*cm',  # 2D
            r'(\d+\.?\d*)\s*cm'  # 1D
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()

                if len(groups) == 3:
                    measurements.append({
                        'dimensions': '3D',
                        'length': float(groups[0]),
                        'width': float(groups[1]),
                        'height': float(groups[2]),
                        'unit': 'cm',
                        'text': match.group(0)
                    })
                elif len(groups) == 2:
                    measurements.append({
                        'dimensions': '2D',
                        'length': float(groups[0]),
                        'width': float(groups[1]),
                        'unit': 'cm',
                        'text': match.group(0)
                    })
                else:
                    measurements.append({
                        'dimensions': '1D',
                        'length': float(groups[0]),
                        'unit': 'cm',
                        'text': match.group(0)
                    })

        return measurements

    async def _extract_response_assessment(
        self,
        context: ProcessingContext
    ) -> Optional[str]:
        """
        Extract tumor response assessment using MedGemma.

        Args:
            context: Processing context

        Returns:
            Response assessment (CR, PR, SD, PD) or None
        """
        prompt = f"""What is the tumor response assessment in this radiology report?

Report text:
{context.raw_text[:1000]}

Response categories:
- CR: Complete Response
- PR: Partial Response
- SD: Stable Disease
- PD: Progressive Disease
- N/A: Not applicable

Return only: CR, PR, SD, PD, or N/A"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=20,
                temperature=0.1
            )

            result = response['text'].strip().upper()

            if result in ['CR', 'PR', 'SD', 'PD']:
                return result

            return None
        except Exception as e:
            self.logger.error(f"Response assessment extraction failed: {e}")
            return None
