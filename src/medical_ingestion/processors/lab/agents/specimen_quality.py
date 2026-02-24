# ============================================================================
# FILE: src/medical_ingestion/processors/lab/agents/specimen_quality.py
# ============================================================================
"""
Specimen Quality Agent - Pre-Analytical Error Detection

Detects specimen quality issues that invalidate results:
- Hemolysis (falsely elevated K+, LDH, AST)
- Lipemia (interference with multiple tests)
- IV contamination (falsely elevated glucose, electrolytes)
- Clotting (falsely decreased platelets, WBC)

Critical for patient safety - prevents treating false values.
"""

from typing import Dict, Any, List, Optional
import logging

from ....core.agent_base import Agent
from ....core.context import ProcessingContext
from ....constants import SPECIMEN_QUALITY_PATTERNS
from ....config import clinical_settings


class SpecimenQualityAgent(Agent):
    """
    Detect pre-analytical errors that invalidate lab results.
    
    Strategy:
    1. Check PDF text for quality indicators ("hemolyzed", "lipemic")
    2. Check for impossible value combinations
    3. Flag or REJECT affected tests
    
    This is the "safety net" that prevents clinical errors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_name(self) -> str:
        return "SpecimenQualityAgent"
    
    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Check specimen quality.
        
        Returns:
            {
                "decision": "accepted" | "rejected" | "flagged",
                "confidence": float,
                "reasoning": str,
                "issues_found": List[str],
                "affected_tests": List[str]
            }
        """
        if not clinical_settings.ENABLE_SPECIMEN_QUALITY:
            return {
                "decision": "skipped",
                "confidence": 1.0,
                "reasoning": "Specimen quality checks disabled",
                "issues_found": [],
                "affected_tests": []
            }
        
        issues_found = []
        affected_tests = []
        
        # Check text for quality indicators
        text = context.raw_text.lower()
        
        for issue_type, pattern_info in SPECIMEN_QUALITY_PATTERNS.items():
            # Skip 'acceptable' status - it's not an issue
            if issue_type == 'acceptable':
                continue

            # Check if any pattern/indicator is present
            # JSON uses 'patterns' key
            patterns = pattern_info.get('patterns', pattern_info.get('indicators', []))

            for pattern in patterns:
                if pattern in text:
                    issues_found.append(issue_type)

                    # Get affected tests (may not exist in all patterns)
                    affected = pattern_info.get('affected_tests', [])
                    if affected:
                        affected_tests.extend(affected)

                    # Get severity and impact info
                    severity = pattern_info.get('severity', 'unknown')
                    impact = pattern_info.get('impact', '')

                    # Flag affected values if we know which tests are affected
                    if affected:
                        for value in context.extracted_values:
                            if value.field_name in affected:
                                value.warnings.append(
                                    f"Specimen quality issue: {issue_type}"
                                )

                    self.logger.warning(
                        f"Specimen quality issue detected: {issue_type} "
                        f"(severity: {severity}, impact: {impact})"
                    )

                    break  # Found this issue type
        
        # Check for impossible combinations (e.g., K+ 7.5 + pH 6.8)
        impossible = self._check_impossible_combinations(context)
        if impossible:
            issues_found.append("impossible_combination")
        
        # Decide if specimen should be rejected
        if issues_found:
            # Check severity from patterns - critical issues mandate rejection
            has_critical = False
            for issue in issues_found:
                if issue in SPECIMEN_QUALITY_PATTERNS:
                    severity = SPECIMEN_QUALITY_PATTERNS[issue].get('severity', '')
                    if severity == 'critical':
                        has_critical = True
                        break
            
            if has_critical:
                # REJECT specimen
                context.specimen_rejected = True
                context.rejection_reason = f"Specimen quality issues: {', '.join(issues_found)}"
                
                self.logger.critical(
                    f"SPECIMEN REJECTED: {context.rejection_reason}"
                )
                
                return {
                    "decision": "rejected",
                    "confidence": 0.95,
                    "reasoning": f"Specimen REJECTED due to: {', '.join(issues_found)}",
                    "issues_found": issues_found,
                    "affected_tests": list(set(affected_tests))
                }
            else:
                # Flag but don't reject
                context.add_quality_flag(
                    f"specimen_quality: {', '.join(issues_found)}",
                    severity="warning"
                )
                
                return {
                    "decision": "flagged",
                    "confidence": 0.75,
                    "reasoning": f"Specimen quality concerns: {', '.join(issues_found)}",
                    "issues_found": issues_found,
                    "affected_tests": list(set(affected_tests))
                }
        
        # No issues found
        return {
            "decision": "accepted",
            "confidence": 0.95,
            "reasoning": "No specimen quality issues detected",
            "issues_found": [],
            "affected_tests": []
        }
    
    def _check_impossible_combinations(self, context: ProcessingContext) -> bool:
        """
        Check for physiologically impossible value combinations.
        
        Example: K+ 7.5 + pH 6.8 = incompatible with life
        
        This catches hemolysis that wasn't marked in the report.
        """
        # Get values by name
        values_dict = {
            v.field_name: v.value
            for v in context.extracted_values
            if v.value is not None
        }
        
        # Check 1: Isolated high K+ with normal Na/Cl suggests hemolysis
        if 'potassium' in values_dict:
            k_val = values_dict['potassium']
            na_val = values_dict.get('sodium')
            cl_val = values_dict.get('chloride')
            
            if k_val > 6.0:  # Critically high potassium
                if na_val and cl_val:
                    # If Na and Cl are normal, likely hemolysis (not true hyperkalemia)
                    if 135 <= na_val <= 145 and 98 <= cl_val <= 106:
                        self.logger.warning(
                            "Isolated high K+ with normal Na/Cl - likely hemolysis"
                        )
                        
                        # Add warning to potassium value
                        for value in context.extracted_values:
                            if value.field_name == 'potassium':
                                value.warnings.append(
                                    "Suspicious isolated elevation - possible hemolysis"
                                )
                        
                        return True
        
        # Check 2: Impossible glucose elevation (>1000 mg/dL)
        if 'glucose' in values_dict:
            gluc_val = values_dict['glucose']
            
            if gluc_val > 1000:
                self.logger.warning(
                    f"Glucose {gluc_val} mg/dL - likely IV contamination or error"
                )
                
                for value in context.extracted_values:
                    if value.field_name == 'glucose':
                        value.warnings.append(
                            "Extreme elevation - verify specimen collection"
                        )
                
                return True
        
        # Check 3: Impossibly low WBC with normal other counts
        if 'wbc' in values_dict:
            wbc_val = values_dict['wbc']
            rbc_val = values_dict.get('rbc')
            plt_val = values_dict.get('platelets')
            
            if wbc_val < 0.5:  # Extremely low
                if rbc_val and plt_val:
                    if rbc_val > 3.0 and plt_val > 100:
                        # Other counts normal - likely clotted specimen
                        self.logger.warning(
                            "Isolated low WBC with normal RBC/PLT - possible clotting"
                        )
                        
                        for value in context.extracted_values:
                            if value.field_name == 'wbc':
                                value.warnings.append(
                                    "Isolated low count - check for specimen clotting"
                                )
                        
                        return True
        
        return False