# ============================================================================
# src/medical_ingestion/processors/radiology/agents/coding_agent.py
# ============================================================================
"""
Medical Coding Agent for Radiology Reports

Maps findings to standard medical codes (ICD-10, CPT, SNOMED).
"""

from typing import List, Dict, Any, Optional
import logging

from medical_ingestion.core.agent_base import Agent
from medical_ingestion.core.context.processing_context import ProcessingContext
from medical_ingestion.medgemma.client import create_client


class CodingAgent(Agent):
    """
    Maps radiology findings to standard medical codes.

    Supports:
    - ICD-10 diagnostic codes
    - CPT procedure codes
    - SNOMED CT codes
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.medgemma = create_client(config)
        self.logger = logging.getLogger(__name__)

        # Common radiology procedure codes
        self.cpt_codes = {
            'chest_xray': '71046',
            'chest_ct': '71250',
            'abdomen_ct': '74150',
            'head_ct': '70450',
            'mri_brain': '70551',
            'ultrasound_abdomen': '76700',
            'mammogram': '77067'
        }

    def get_name(self) -> str:
        return "CodingAgent"

    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Assign medical codes to radiology report.

        Args:
            context: Processing context with findings

        Returns:
            Dict with assigned codes
        """
        self.logger.info("Assigning medical codes to radiology report")

        # Determine procedure type
        procedure_type = self._detect_procedure_type(context.raw_text)

        # Get CPT code for procedure
        cpt_code = self._get_cpt_code(procedure_type)

        # Get ICD-10 codes for findings
        icd10_codes = await self._get_icd10_codes(context)

        # Get SNOMED codes for findings
        snomed_codes = await self._get_snomed_codes(context)

        return {
            'procedure_type': procedure_type,
            'cpt_code': cpt_code,
            'icd10_codes': icd10_codes,
            'snomed_codes': snomed_codes
        }

    def _detect_procedure_type(self, text: str) -> str:
        """
        Detect imaging procedure type from report text.

        Args:
            text: Report text

        Returns:
            Procedure type
        """
        text_lower = text.lower()

        if 'ct' in text_lower or 'computed tomography' in text_lower:
            if 'chest' in text_lower:
                return 'chest_ct'
            elif 'abdomen' in text_lower:
                return 'abdomen_ct'
            elif 'head' in text_lower or 'brain' in text_lower:
                return 'head_ct'
            return 'ct_unknown'

        elif 'mri' in text_lower or 'magnetic resonance' in text_lower:
            if 'brain' in text_lower or 'head' in text_lower:
                return 'mri_brain'
            return 'mri_unknown'

        elif 'ultrasound' in text_lower or 'sonogram' in text_lower:
            if 'abdomen' in text_lower:
                return 'ultrasound_abdomen'
            return 'ultrasound_unknown'

        elif 'mammogram' in text_lower or 'mammography' in text_lower:
            return 'mammogram'

        elif 'x-ray' in text_lower or 'radiograph' in text_lower:
            if 'chest' in text_lower:
                return 'chest_xray'
            return 'xray_unknown'

        return 'unknown'

    def _get_cpt_code(self, procedure_type: str) -> Optional[str]:
        """
        Get CPT code for procedure type.

        Args:
            procedure_type: Type of imaging procedure

        Returns:
            CPT code or None
        """
        return self.cpt_codes.get(procedure_type)

    async def _get_icd10_codes(
        self,
        context: ProcessingContext
    ) -> List[Dict[str, str]]:
        """
        Get ICD-10 codes for findings using MedGemma.

        Args:
            context: Processing context

        Returns:
            List of ICD-10 codes with descriptions
        """
        # Get impression/findings
        impression = context.clinical_summary or context.raw_text[:500]

        prompt = f"""Assign appropriate ICD-10 codes for this radiology finding.

Finding:
{impression}

Return format:
CODE: Description

Example:
J18.9: Pneumonia, unspecified organism"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )

            # Parse codes
            codes = []
            for line in response['text'].split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    code = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else ''

                    codes.append({
                        'code': code,
                        'description': description,
                        'system': 'ICD-10'
                    })

            return codes
        except Exception as e:
            self.logger.error(f"ICD-10 coding failed: {e}")
            return []

    async def _get_snomed_codes(
        self,
        context: ProcessingContext
    ) -> List[Dict[str, str]]:
        """
        Get SNOMED CT codes for findings using MedGemma.

        Args:
            context: Processing context

        Returns:
            List of SNOMED codes with descriptions
        """
        # Get impression/findings
        impression = context.clinical_summary or context.raw_text[:500]

        prompt = f"""Assign appropriate SNOMED CT codes for this radiology finding.

Finding:
{impression}

Return format:
CODE: Description

Example:
233604007: Pneumonia"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )

            # Parse codes
            codes = []
            for line in response['text'].split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    code = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else ''

                    codes.append({
                        'code': code,
                        'description': description,
                        'system': 'SNOMED-CT'
                    })

            return codes
        except Exception as e:
            self.logger.error(f"SNOMED coding failed: {e}")
            return []
