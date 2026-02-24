# ============================================================================
# FILE 3: src/medical_ingestion/temporal/significance.py
# ============================================================================
"""
Clinical Significance Assessor

Determines clinical significance of detected trends and patterns.
Provides:
- Severity assessment (critical, high, medium, low)
- Clinical interpretation
- Recommended actions
- Urgency level
"""

from typing import Dict, Any, Optional
import logging


class ClinicalSignificanceAssessor:
    """
    Assesses clinical significance of temporal patterns.
    
    Translates statistical patterns into clinical recommendations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.significance_rules = self._load_significance_rules()
    
    def _load_significance_rules(self) -> Dict[str, Dict]:
        """
        Load clinical significance rules for different tests.
        
        Each test has thresholds and interpretations.
        """
        return {
            'hemoglobin': {
                'acute_drop': {
                    '>3.0': {'severity': 'critical', 'interpretation': 'Possible acute hemorrhage', 'action': 'Urgent evaluation'},
                    '>2.0': {'severity': 'high', 'interpretation': 'Significant blood loss', 'action': 'Prompt evaluation'},
                },
                'chronic_decline': {
                    'any': {'severity': 'high', 'interpretation': 'Progressive anemia', 'action': 'Investigate etiology'},
                }
            },
            'creatinine': {
                'acute_rise': {
                    '>0.5': {'severity': 'critical', 'interpretation': 'Acute kidney injury', 'action': 'Immediate nephrology consult'},
                },
                'chronic_rise': {
                    'any': {'severity': 'high', 'interpretation': 'Progressive kidney disease', 'action': 'Nephrology referral'},
                }
            },
            'potassium': {
                'value': {
                    '>6.5': {'severity': 'critical', 'interpretation': 'Life-threatening hyperkalemia', 'action': 'Immediate treatment'},
                    '<2.5': {'severity': 'critical', 'interpretation': 'Severe hypokalemia', 'action': 'Immediate treatment'},
                }
            },
            'glucose': {
                'chronic_elevation': {
                    'any': {'severity': 'medium', 'interpretation': 'Poor glycemic control', 'action': 'Adjust diabetes management'},
                },
                'sawtooth': {
                    'any': {'severity': 'medium', 'interpretation': 'Glycemic variability', 'action': 'Review medication adherence'},
                }
            },
            'inr': {
                'sawtooth': {
                    'any': {'severity': 'high', 'interpretation': 'Unstable anticoagulation', 'action': 'Review warfarin adherence'},
                }
            }
        }
    
    def assess_significance(
        self,
        pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess clinical significance of a detected pattern.
        
        Args:
            pattern: Pattern dict from PatternAnalyzer
            
        Returns:
            {
                "severity": str,
                "interpretation": str,
                "recommended_action": str,
                "urgency": str,
                "reasoning": str
            }
        """
        test_name = pattern['test']
        pattern_type = pattern['type']
        
        # Get rules for this test
        if test_name not in self.significance_rules:
            return self._default_assessment(pattern)
        
        rules = self.significance_rules[test_name]
        
        # Apply rules based on pattern type
        if pattern_type in rules:
            rule_set = rules[pattern_type]
            return self._apply_rules(pattern, rule_set)
        
        return self._default_assessment(pattern)
    
    def _apply_rules(
        self,
        pattern: Dict[str, Any],
        rule_set: Dict
    ) -> Dict[str, Any]:
        """Apply significance rules to pattern"""
        
        # For patterns with magnitude (acute_drop, acute_rise)
        if 'change' in pattern:
            magnitude = abs(pattern['change'])
            
            # Check each threshold
            for threshold_str, rule in sorted(rule_set.items(), reverse=True):
                if threshold_str == 'any':
                    continue
                
                threshold = float(threshold_str.strip('>'))
                if magnitude > threshold:
                    return {
                        'severity': rule['severity'],
                        'interpretation': rule['interpretation'],
                        'recommended_action': rule['action'],
                        'urgency': self._severity_to_urgency(rule['severity']),
                        'reasoning': f"Change of {magnitude:.1f} exceeds {threshold} threshold"
                    }
            
            # Check 'any' rule
            if 'any' in rule_set:
                rule = rule_set['any']
                return {
                    'severity': rule['severity'],
                    'interpretation': rule['interpretation'],
                    'recommended_action': rule['action'],
                    'urgency': self._severity_to_urgency(rule['severity']),
                    'reasoning': f"Pattern detected: {pattern['type']}"
                }
        
        # For patterns without magnitude
        if 'any' in rule_set:
            rule = rule_set['any']
            return {
                'severity': rule['severity'],
                'interpretation': rule['interpretation'],
                'recommended_action': rule['action'],
                'urgency': self._severity_to_urgency(rule['severity']),
                'reasoning': f"Pattern detected: {pattern['type']}"
            }
        
        return self._default_assessment(pattern)
    
    def _default_assessment(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Default assessment when no specific rules apply"""
        severity = pattern.get('severity', 'medium')
        
        return {
            'severity': severity,
            'interpretation': f"Temporal pattern detected in {pattern['test']}",
            'recommended_action': "Review trend and clinical context",
            'urgency': self._severity_to_urgency(severity),
            'reasoning': pattern.get('description', 'Pattern identified')
        }
    
    def _severity_to_urgency(self, severity: str) -> str:
        """Convert severity to urgency level"""
        mapping = {
            'critical': 'immediate',
            'high': 'urgent',
            'medium': 'routine',
            'low': 'routine'
        }
        return mapping.get(severity, 'routine')