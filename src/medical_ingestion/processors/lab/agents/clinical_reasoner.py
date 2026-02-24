# ============================================================================
# src/medical_ingestion/processors/lab/agents/clinical_reasoner.py
# ============================================================================
"""
Clinical Reasoning Agent - Summaries & Reflex Protocols

Generates:
1. 3-sentence clinical summary (for busy physicians)
2. Reflex test recommendations (accelerate diagnosis)
3. Follow-up plan (urgent vs routine)

Uses MedGemma for medical reasoning + guideline protocols.
"""

from typing import Dict, Any, List
import logging

from ....core.agent_base import Agent
from ....core.context import ProcessingContext
from ....medgemma.client import create_client
from ....constants import REFLEX_PROTOCOLS, CRITICAL_VALUES
from ....config import clinical_settings


class ClinicalReasoningAgent(Agent):
    """
    Generate clinical intelligence from lab results.
    
    Outputs:
    - Clinical summary (3 sentences, actionable)
    - Reflex test recommendations (with guideline citations)
    - Urgency assessment (urgent vs routine follow-up)
    
    This is where data becomes clinical intelligence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.medgemma = create_client(config)
    
    def get_name(self) -> str:
        return "ClinicalReasoningAgent"
    
    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Generate clinical reasoning.
        
        Returns:
            {
                "decision": "reasoning_complete",
                "confidence": float,
                "reasoning": str,
                "summary_generated": bool,
                "reflex_tests": int
            }
        """
        if not context.extracted_values:
            return {
                "decision": "skipped",
                "confidence": 1.0,
                "reasoning": "No values to reason about",
                "summary_generated": False,
                "reflex_tests": 0
            }
        
        # 1. Apply reflex protocols (rule-based)
        reflex_count = 0
        if clinical_settings.ENABLE_REFLEX_PROTOCOLS:
            reflex_count = self._apply_reflex_protocols(context)
        
        # 2. Generate clinical summary (AI-based)
        summary_generated = False
        if not context.specimen_rejected:  # Don't summarize rejected specimens
            summary = await self._generate_clinical_summary(context)
            if summary:
                context.clinical_summary = summary
                summary_generated = True
        
        return {
            "decision": "reasoning_complete",
            "confidence": 0.85,
            "reasoning": f"Generated summary, {reflex_count} reflex tests recommended",
            "summary_generated": summary_generated,
            "reflex_tests": reflex_count
        }
    
    def _apply_reflex_protocols(self, context: ProcessingContext) -> int:
        """
        Apply guideline-based reflex testing protocols.
        
        Example: TSH elevated → order Free T4 + TPO antibodies
        
        Returns count of reflex tests recommended.
        """
        reflex_count = 0
        
        # Get values as dict
        values_dict = {
            v.field_name: v.value
            for v in context.extracted_values
            if v.value is not None
        }
        
        # Get patient demographics
        age = context.patient_demographics.get('age')
        sex = context.patient_demographics.get('sex')
        
        # Check each protocol
        for protocol_name, protocol in REFLEX_PROTOCOLS.items():
            condition_func = protocol['condition']
            
            # Check if condition is met
            try:
                # Different protocols have different parameters
                if protocol_name in ['tsh_elevated', 'tsh_suppressed']:
                    if 'tsh' in values_dict:
                        if condition_func(values_dict['tsh']):
                            self._add_reflex_recommendation(
                                context, protocol_name, protocol
                            )
                            reflex_count += 1
                
                elif protocol_name == 'anemia':
                    if 'hemoglobin' in values_dict and sex:
                        if condition_func(values_dict['hemoglobin'], sex):
                            self._add_reflex_recommendation(
                                context, protocol_name, protocol
                            )
                            reflex_count += 1
                
                elif protocol_name == 'prediabetes':
                    if 'glucose' in values_dict:
                        if condition_func(values_dict['glucose']):
                            self._add_reflex_recommendation(
                                context, protocol_name, protocol
                            )
                            reflex_count += 1
            
            except Exception as e:
                self.logger.error(f"Error checking protocol {protocol_name}: {e}")
        
        return reflex_count
    
    def _add_reflex_recommendation(
        self,
        context: ProcessingContext,
        protocol_name: str,
        protocol: Dict
    ):
        """Add reflex test recommendation to context"""
        recommendation = {
            "protocol": protocol_name,
            "tests": protocol['reflex_tests'],
            "reasoning": protocol['reasoning'],
            "guideline": protocol['guideline']
        }
        
        context.reflex_recommendations.append(recommendation)
        
        self.logger.info(
            f"Reflex protocol triggered: {protocol_name} → "
            f"{', '.join(protocol['reflex_tests'])}"
        )
    
    async def _generate_clinical_summary(self, context: ProcessingContext) -> str:
        """
        Generate 3-sentence clinical summary using MedGemma.
        
        Focuses on:
        1. Key abnormalities
        2. Clinical significance
        3. Recommended actions
        """
        # Prepare lab results for prompt
        results_text = []
        for value in context.extracted_values[:10]:  # Limit to 10 for brevity
            abnormal = ""
            if value.abnormal_flag:
                abnormal = f" [{value.abnormal_flag}]"
            
            results_text.append(
                f"{value.field_name}: {value.value} {value.unit}{abnormal}"
            )
        
        # Include temporal trends if available
        trends_text = ""
        if context.temporal_trends:
            trends = [t['description'] for t in context.temporal_trends[:3]]
            trends_text = "\nTrends: " + "; ".join(trends)
        
        # Include quality flags
        flags_text = ""
        if context.quality_flags:
            flags_text = "\nQuality flags: " + ", ".join(context.quality_flags)
        
        prompt = f"""Generate a concise 3-sentence clinical summary of these lab results.

Patient: {context.patient_demographics.get('age', 'unknown')} years, {context.patient_demographics.get('sex', 'unknown')}

Results:
{chr(10).join(results_text)}
{trends_text}
{flags_text}

Write EXACTLY 3 sentences that:
1. Identify key abnormalities
2. Explain clinical significance
3. Recommend urgent vs routine follow-up

Summary:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.2
            )
            
            # Extract summary (first 3 sentences)
            summary = response['text'].strip()
            
            # Ensure it's concise (limit to 3 sentences)
            sentences = summary.split('.')
            summary = '. '.join(sentences[:3]) + '.'
            
            self.logger.info("Clinical summary generated")
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return ""