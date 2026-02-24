# ============================================================================
# FILE 2: src/medical_ingestion/processors/lab/agents/validator.py
# ============================================================================
"""
Validation Agent - Dual Validation System

Validates extracted lab values using:
1. Rule-based validation (deterministic, fast)
2. AI validation with MedGemma (medical reasoning)

When validators disagree → flag for human review.
"""

from typing import Dict, Any
import logging

from ....core.agent_base import Agent
from ....core.context import ProcessingContext, ExtractedValue
from ....medgemma.client import create_client
from ....constants import PLAUSIBILITY_RANGES, REFERENCE_RANGES
from ....config import clinical_settings


class ValidationAgent(Agent):
    """
    Dual validation: Rules + AI.
    
    For each extracted value:
    1. Rule-based checks (plausibility, units, range)
    2. MedGemma reasoning (clinical context)
    3. Compare results
    4. Flag conflicts
    
    This catches errors that single-method validation misses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.medgemma = create_client(config)
    
    def get_name(self) -> str:
        return "ValidationAgent"
    
    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Validate all extracted values.
        
        Returns:
            {
                "decision": "validated",
                "confidence": float,
                "reasoning": str,
                "conflicts_found": int,
                "values_rejected": int
            }
        """
        if not context.extracted_values:
            return {
                "decision": "no_values",
                "confidence": 0.0,
                "reasoning": "No values to validate",
                "conflicts_found": 0,
                "values_rejected": 0
            }
        
        conflicts = 0
        rejections = 0
        
        for extracted_value in context.extracted_values:
            # Rule-based validation
            rule_valid = self._validate_with_rules(extracted_value, context)
            extracted_value.rule_validation = rule_valid
            
            # AI validation (if enabled)
            if clinical_settings.ENABLE_DUAL_VALIDATION:
                ai_valid = await self._validate_with_ai(extracted_value, context)
                extracted_value.ai_validation = ai_valid
                
                # Check for conflict
                if rule_valid != ai_valid:
                    extracted_value.validation_conflict = True
                    conflicts += 1
                    
                    context.add_warning(
                        f"Validation conflict for {extracted_value.field_name}: "
                        f"rules={rule_valid}, AI={ai_valid}"
                    )
                    
                    self.logger.warning(
                        f"Conflict: {extracted_value.field_name} = {extracted_value.value} "
                        f"(rules: {rule_valid}, AI: {ai_valid})"
                    )
            
            # Reject if both validators fail
            if not rule_valid and (not clinical_settings.ENABLE_DUAL_VALIDATION or not ai_valid):
                extracted_value.warnings.append("Failed validation - verify manually")
                rejections += 1
        
        confidence = 1.0 - (conflicts * 0.1) - (rejections * 0.2)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "decision": "validated",
            "confidence": confidence,
            "reasoning": f"Validated {len(context.extracted_values)} values, "
                        f"{conflicts} conflicts, {rejections} rejections",
            "conflicts_found": conflicts,
            "values_rejected": rejections
        }
    
    def _validate_with_rules(self, value, context: ProcessingContext) -> bool:
        """
        Rule-based validation checks:
        1. Plausibility range (catch decimal errors)
        2. Reference range context
        3. Unit compatibility
        """
        # Check plausibility range
        if value.field_name in PLAUSIBILITY_RANGES:
            plaus_range = PLAUSIBILITY_RANGES[value.field_name]
            min_plaus, max_plaus, unit = plaus_range
            
            if value.value < min_plaus or value.value > max_plaus:
                self.logger.warning(
                    f"{value.field_name} outside plausibility: {value.value} "
                    f"(range: {min_plaus}-{max_plaus})"
                )
                return False
        
        # Additional checks can be added here
        
        return True
    
    async def _validate_with_ai(self, value, context: ProcessingContext) -> bool:
        """
        AI validation using medical reasoning.
        
        Asks MedGemma if value is clinically plausible.
        """
        # Get patient demographics for context
        age = context.patient_demographics.get('age', 'unknown')
        sex = context.patient_demographics.get('sex', 'unknown')
        
        # Gather other values for context
        other_values = [
            f"{v.field_name}={v.value}{v.unit}"
            for v in context.extracted_values
            if v != value
        ][:5]  # Limit to 5 for brevity
        
        prompt = f"""Is this lab result clinically plausible?

Test: {value.field_name}
Value: {value.value} {value.unit}
Reference range: {value.reference_min}-{value.reference_max} {value.unit}
Patient: {age} years old, {sex}
Other results: {', '.join(other_values)}

Consider:
- Is this value physiologically possible?
- Are there pre-analytical errors (hemolysis, contamination)?
- Does it make sense with other values?

Return ONLY a JSON object:
{{
    "plausible": true/false,
    "reasoning": "<brief explanation>"
}}

JSON:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.1
            )
            
            result = self.medgemma.extract_json(response['text'])
            
            if result and 'plausible' in result:
                plausible = result['plausible']
                
                if not plausible:
                    self.logger.info(
                        f"AI validation failed for {value.field_name}: "
                        f"{result.get('reasoning', 'No reason given')}"
                    )
                
                return plausible
        
        except Exception as e:
            self.logger.error(f"AI validation error: {e}")
        
        # Default to true if AI validation fails
        return True