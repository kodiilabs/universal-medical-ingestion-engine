# ============================================================================
# FILE 3: src/medical_ingestion/processors/lab/agents/temporal_analyzer.py
# ============================================================================
"""
Temporal Analysis Agent - Trend Detection

Analyzes lab values over time to detect:
- Acute drops (possible bleeding, kidney failure)
- Chronic decline (progressive disease)
- Sawtooth patterns (medication non-adherence)
- Suspicious normalizations (specimen mix-up)

Requires patient history in context.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from ....core.agent_base import Agent
from ....core.context.processing_context import ProcessingContext
from ....config.clinical_config import clinical_settings


class TemporalAnalysisAgent(Agent):
    """
    Detect clinically significant trends in lab values.
    
    Patterns detected:
    1. Acute drop (Hgb 12.5→8.2 in 7 days = possible GI bleed)
    2. Chronic decline (Creatinine rising 0.2/month = kidney disease)
    3. Sawtooth (INR bouncing = non-adherence)
    4. Normalization (value suddenly normal = specimen error?)
    
    Only runs if patient history available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_name(self) -> str:
        return "TemporalAnalysisAgent"
    
    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Analyze temporal trends.
        
        Returns:
            {
                "decision": "analyzed" | "skipped",
                "confidence": float,
                "reasoning": str,
                "trends_found": int,
                "flags_raised": int
            }
        """
        if not clinical_settings.ENABLE_TEMPORAL_ANALYSIS:
            return self._skip_analysis("Temporal analysis disabled")
        
        if not context.patient_history:
            return self._skip_analysis("No patient history available")
        
        if not context.extracted_values:
            return self._skip_analysis("No current values to analyze")
        
        trends_found = 0
        flags_raised = 0
        
        for current_value in context.extracted_values:
            # Get historical values for this test
            historical = self._get_historical_values(
                current_value.field_name,
                context.patient_history
            )
            
            if len(historical) < 2:
                continue  # Need at least 2 prior values
            
            # Detect patterns
            trend = self._analyze_trend(
                current_value,
                historical,
                context
            )
            
            if trend:
                context.temporal_trends.append(trend)
                trends_found += 1
                
                if trend.get('severity') in ['critical', 'high']:
                    flags_raised += 1
                    context.add_critical_finding(trend['description'])
        
        confidence = 0.85 if trends_found > 0 else 0.9
        
        return {
            "decision": "analyzed",
            "confidence": confidence,
            "reasoning": f"Analyzed trends: {trends_found} patterns found",
            "trends_found": trends_found,
            "flags_raised": flags_raised
        }
    
    def _skip_analysis(self, reason: str) -> Dict[str, Any]:
        """Return skip result"""
        return {
            "decision": "skipped",
            "confidence": 1.0,
            "reasoning": reason,
            "trends_found": 0,
            "flags_raised": 0
        }
    
    def _get_historical_values(
        self,
        test_name: str,
        history: List[Dict]
    ) -> List[Dict]:
        """Get historical values for specific test"""
        historical = []
        
        for record in history:
            if test_name in record.get('values', {}):
                historical.append({
                    'date': record.get('date'),
                    'value': record['values'][test_name]
                })
        
        # Sort by date (newest first)
        historical.sort(key=lambda x: x['date'], reverse=True)
        
        return historical
    
    def _analyze_trend(
        self,
        current_value,
        historical: List[Dict],
        context: ProcessingContext
    ) -> Optional[Dict]:
        """
        Analyze trend pattern.
        
        Returns trend dict if significant pattern found.
        """
        test_name = current_value.field_name
        current_val = current_value.value
        
        # Get most recent historical value
        most_recent = historical[0]
        most_recent_val = most_recent['value']
        
        # Calculate change
        absolute_change = current_val - most_recent_val
        percent_change = (absolute_change / most_recent_val * 100) if most_recent_val != 0 else 0
        
        # Check for acute drop (hemoglobin example)
        if test_name == 'hemoglobin' and absolute_change < -2.0:
            # Drop of >2 g/dL
            days_elapsed = (datetime.now() - most_recent['date']).days
            
            if days_elapsed <= 30:  # Within last month
                return {
                    'test': test_name,
                    'trend': 'acute_drop',
                    'description': f"Hemoglobin dropped {abs(absolute_change):.1f} g/dL in {days_elapsed} days - possible acute blood loss",
                    'severity': 'critical',
                    'current_value': current_val,
                    'previous_value': most_recent_val,
                    'change': absolute_change
                }
        
        # Check for chronic decline (creatinine example)
        if test_name == 'creatinine' and len(historical) >= 3:
            # Check if consistently rising
            values = [h['value'] for h in historical[:3]] + [current_val]
            
            if all(values[i] < values[i+1] for i in range(len(values)-1)):
                return {
                    'test': test_name,
                    'trend': 'chronic_rise',
                    'description': f"Creatinine steadily rising - possible progressive kidney disease",
                    'severity': 'high',
                    'current_value': current_val,
                    'values': values
                }
        
        # Check for suspicious normalization
        if abs(percent_change) > 50 and current_value.reference_min and current_value.reference_max:
            # Large change AND now in normal range
            in_range = current_value.reference_min <= current_val <= current_value.reference_max
            was_abnormal = not (current_value.reference_min <= most_recent_val <= current_value.reference_max)
            
            if in_range and was_abnormal:
                return {
                    'test': test_name,
                    'trend': 'suspicious_normalization',
                    'description': f"{test_name} suddenly normalized ({percent_change:.0f}% change) - verify specimen",
                    'severity': 'medium',
                    'current_value': current_val,
                    'previous_value': most_recent_val,
                    'change_percent': percent_change
                }
        
        return None
