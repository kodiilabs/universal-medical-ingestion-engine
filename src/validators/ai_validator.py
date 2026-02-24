# ============================================================================
# FILE: src/validators/ai_validator.py
# ============================================================================
"""
AI Validator using MedGemma

Validates extracted values using medical reasoning.
Catches errors that rules miss:
- Pre-analytical issues (hemolysis, contamination)
- Clinical implausibility (doesn't fit patient context)
- Lab-specific quirks

Slower than rule validation but more context-aware.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_ingestion.medgemma.client import create_client
from medical_ingestion.core.context.extracted_value import ExtractedValue
from medical_ingestion.core.context.processing_context import ProcessingContext
from medical_ingestion.config import medgemma_settings


logger = logging.getLogger(__name__)


class AIValidator:
    """
    AI-powered validation using MedGemma.

    Asks the model:
    - Is this value clinically plausible?
    - Does it fit with patient demographics?
    - Are there pre-analytical errors?
    - Does it correlate with other values?
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.medgemma = create_client(self.config)

    async def validate(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext
    ) -> Tuple[bool, str, float]:
        """
        Validate single value with AI reasoning.

        Args:
            extracted: Value to validate
            context: Processing context with patient info

        Returns:
            (is_valid, reasoning, confidence)
        """
        prompt = self._build_validation_prompt(extracted, context)

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1  # Low temperature for deterministic validation
            )

            result = self._parse_response(response['text'])

            if result:
                is_valid = result['plausible']
                reasoning = result['reasoning']
                confidence = result.get('confidence', 0.8)

                logger.info(
                    f"AI validation for {extracted.field_name}: "
                    f"valid={is_valid}, reasoning={reasoning}"
                )

                return is_valid, reasoning, confidence

        except Exception as e:
            logger.error(f"AI validation error: {e}")

        # Default to valid if AI fails (don't block on AI errors)
        return True, "AI validation failed - defaulting to valid", 0.5

    def _build_validation_prompt(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext
    ) -> str:
        """
        Build prompt for MedGemma validation.
        """
        # Get patient demographics
        age = context.patient_demographics.get('age', 'unknown')
        sex = context.patient_demographics.get('sex', 'unknown')

        # Get other lab values for context (limit to 5)
        other_values = []
        for v in context.extracted_values[:5]:
            if v != extracted and v.value is not None:
                other_values.append(f"{v.field_name}={v.value}{v.unit}")

        other_values_str = ", ".join(other_values) if other_values else "none"

        # Build prompt
        prompt = f"""You are validating a lab result for clinical plausibility.

Test: {extracted.field_name}
Value: {extracted.value} {extracted.unit}
Reference range: {extracted.reference_min}-{extracted.reference_max} {extracted.unit}

Patient information:
- Age: {age} years
- Sex: {sex}

Other lab results from same panel:
{other_values_str}

Evaluate this result for:
1. Physiological plausibility (is this value possible?)
2. Pre-analytical errors (hemolysis, contamination, timing issues)
3. Clinical context (does it fit with other values?)
4. Common lab errors (decimal points, units)

Return ONLY a JSON object (no other text):
{{
    "plausible": true or false,
    "reasoning": "brief explanation (max 50 words)",
    "confidence": 0.0-1.0
}}

JSON:"""

        return prompt

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from MedGemma.
        """
        try:
            # Try to extract JSON from response
            # MedGemma sometimes adds extra text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)

                # Validate required fields
                if 'plausible' in result and 'reasoning' in result:
                    return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response: {e}")

        return None

    async def batch_validate(
        self,
        context: ProcessingContext
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate all extracted values in context.

        Returns:
            {field_name: {
                "valid": bool,
                "reasoning": str,
                "confidence": float
            }}
        """
        results = {}

        for extracted in context.extracted_values:
            is_valid, reasoning, confidence = await self.validate(extracted, context)

            results[extracted.field_name] = {
                "valid": is_valid,
                "reasoning": reasoning,
                "confidence": confidence
            }

            # Update extracted value
            extracted.ai_validation = is_valid

        return results

    async def validate_with_explanation(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """
        Validate and return detailed explanation.

        Useful for debugging or human review.
        """
        is_valid, reasoning, confidence = await self.validate(extracted, context)

        return {
            "field_name": extracted.field_name,
            "value": extracted.value,
            "unit": extracted.unit,
            "valid": is_valid,
            "reasoning": reasoning,
            "confidence": confidence,
            "reference_range": f"{extracted.reference_min}-{extracted.reference_max}",
            "rule_validation": extracted.rule_validation
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def validate_with_ai(
    extracted: ExtractedValue,
    context: ProcessingContext
) -> bool:
    """
    Quick AI validation check.

    Returns:
        True if valid, False otherwise
    """
    validator = AIValidator()
    is_valid, _, _ = await validator.validate(extracted, context)
    return is_valid
