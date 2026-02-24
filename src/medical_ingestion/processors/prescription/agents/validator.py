# ============================================================================
# src/medical_ingestion/processors/prescription/agents/validator.py
# ============================================================================
"""
Prescription Validation Agent

Validates prescription data for:
- Required field completeness
- Dosage safety
- Drug-drug interactions
- Drug-allergy conflicts
- Duplicate therapy
- Controlled substance regulations
"""

from typing import Dict, Any, List
import logging
import json
from pathlib import Path

from medical_ingestion.core.agent_base import Agent
from medical_ingestion.core.context.processing_context import ProcessingContext
from medical_ingestion.medgemma.client import create_client


class PrescriptionValidator(Agent):
    """
    Validates prescription data for safety and completeness.

    Checks:
    - Required fields (drug name, dosage, route, frequency)
    - Dosage within safe limits
    - Drug-drug interactions
    - Drug-allergy conflicts
    - Duplicate therapy warnings
    - Controlled substance compliance
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.medgemma = create_client(config)
        self.logger = logging.getLogger(__name__)

        # Load RxNorm mappings for drug name validation
        knowledge_dir = Path(__file__).parent.parent.parent.parent.parent / "knowledge"
        rxnorm_path = knowledge_dir / "rxnorm_mappings.json"

        if rxnorm_path.exists():
            with open(rxnorm_path) as f:
                self.rxnorm_data = json.load(f)
        else:
            self.rxnorm_data = {}

        # Controlled substance schedules
        self.controlled_schedules = ['Schedule II', 'Schedule III', 'Schedule IV', 'Schedule V']

    def get_name(self) -> str:
        return "PrescriptionValidator"

    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Validate prescription data.

        Args:
            context: Processing context with prescription data

        Returns:
            Dict with validation results and warnings
        """
        self.logger.info("Validating prescription data")

        medications = context.metadata.get('medications', [])

        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'completeness_check': {},
            'safety_check': {},
            'interaction_check': {},
            'allergy_check': {}
        }

        # Validate each medication
        for i, medication in enumerate(medications):
            med_prefix = f"Medication {i+1}"

            # Check required fields
            completeness = self._check_completeness(medication)
            validation_results['completeness_check'][med_prefix] = completeness

            if not completeness['is_complete']:
                validation_results['is_valid'] = False
                validation_results['errors'].extend([
                    f"{med_prefix}: Missing {field}"
                    for field in completeness['missing_fields']
                ])

            # Check dosage safety
            safety = await self._check_dosage_safety(medication)
            validation_results['safety_check'][med_prefix] = safety

            if safety.get('unsafe'):
                validation_results['warnings'].append(
                    f"{med_prefix}: {safety.get('reason', 'Dosage concern')}"
                )

            # Check for controlled substance
            if self._is_controlled_substance(medication):
                controlled_check = self._check_controlled_substance_compliance(medication, context)
                if not controlled_check['compliant']:
                    validation_results['warnings'].extend([
                        f"{med_prefix}: {warning}"
                        for warning in controlled_check['warnings']
                    ])

        # Check drug-drug interactions
        if len(medications) > 1:
            interactions = await self._check_drug_interactions(medications)
            validation_results['interaction_check'] = interactions

            if interactions.get('has_interactions'):
                validation_results['warnings'].extend([
                    f"Drug interaction: {interaction}"
                    for interaction in interactions.get('interactions', [])
                ])

        # Check drug-allergy conflicts
        patient = context.metadata.get('patient', {})
        allergies = patient.get('allergies', [])

        if allergies:
            allergy_conflicts = await self._check_allergy_conflicts(medications, allergies)
            validation_results['allergy_check'] = allergy_conflicts

            if allergy_conflicts.get('has_conflicts'):
                validation_results['is_valid'] = False
                validation_results['errors'].extend([
                    f"Allergy conflict: {conflict}"
                    for conflict in allergy_conflicts.get('conflicts', [])
                ])

        # Check for duplicate therapy
        duplicates = self._check_duplicate_therapy(medications)
        if duplicates:
            validation_results['warnings'].extend([
                f"Potential duplicate therapy: {dup}"
                for dup in duplicates
            ])

        self.logger.info(
            f"Validation complete: {'VALID' if validation_results['is_valid'] else 'INVALID'} "
            f"({len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings)"
        )

        return validation_results

    def _check_completeness(self, medication: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if medication has all required fields.

        Args:
            medication: Medication dict

        Returns:
            Dict with completeness check results
        """
        required_fields = [
            'medication_name',
            'strength/dosage',
            'route',
            'frequency'
        ]

        missing_fields = []
        for field in required_fields:
            if not medication.get(field):
                missing_fields.append(field)

        return {
            'is_complete': len(missing_fields) == 0,
            'missing_fields': missing_fields,
            'required_fields': required_fields
        }

    async def _check_dosage_safety(
        self,
        medication: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if dosage is within safe limits using MedGemma.

        Args:
            medication: Medication dict

        Returns:
            Dict with safety check results
        """
        drug_name = medication.get('medication_name', '')
        dosage = medication.get('strength/dosage', '')
        frequency = medication.get('frequency', '')

        prompt = f"""Is this dosage safe and appropriate?

Medication: {drug_name}
Dosage: {dosage}
Frequency: {frequency}

Answer: SAFE or UNSAFE
If UNSAFE, explain why in 1 sentence."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1
            )

            result_text = response['text'].strip()

            if 'UNSAFE' in result_text.upper():
                return {
                    'safe': False,
                    'unsafe': True,
                    'reason': result_text.replace('UNSAFE', '').strip()
                }

            return {'safe': True, 'unsafe': False}
        except Exception as e:
            self.logger.error(f"Dosage safety check failed: {e}")
            return {'safe': None, 'error': str(e)}

    async def _check_drug_interactions(
        self,
        medications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check for drug-drug interactions using MedGemma.

        Args:
            medications: List of medication dicts

        Returns:
            Dict with interaction check results
        """
        drug_names = [med.get('medication_name', '') for med in medications]

        prompt = f"""Check for drug-drug interactions between these medications:

{', '.join(drug_names)}

List any significant interactions.
Return "None" if no significant interactions."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.1
            )

            result_text = response['text'].strip()

            if 'none' in result_text.lower():
                return {
                    'has_interactions': False,
                    'interactions': []
                }

            # Parse interactions
            interactions = []
            for line in result_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•')):
                    interactions.append(line.lstrip('- •').strip())

            return {
                'has_interactions': len(interactions) > 0,
                'interactions': interactions
            }
        except Exception as e:
            self.logger.error(f"Drug interaction check failed: {e}")
            return {'has_interactions': None, 'error': str(e)}

    async def _check_allergy_conflicts(
        self,
        medications: List[Dict[str, Any]],
        allergies: List[str]
    ) -> Dict[str, Any]:
        """
        Check for drug-allergy conflicts using MedGemma.

        Args:
            medications: List of medication dicts
            allergies: List of patient allergies

        Returns:
            Dict with allergy conflict results
        """
        drug_names = [med.get('medication_name', '') for med in medications]

        prompt = f"""Check if any of these medications conflict with the patient's allergies:

Medications: {', '.join(drug_names)}
Allergies: {', '.join(allergies)}

List any conflicts or cross-sensitivities.
Return "None" if no conflicts."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=250,
                temperature=0.1
            )

            result_text = response['text'].strip()

            if 'none' in result_text.lower():
                return {
                    'has_conflicts': False,
                    'conflicts': []
                }

            # Parse conflicts
            conflicts = []
            for line in result_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•')):
                    conflicts.append(line.lstrip('- •').strip())

            return {
                'has_conflicts': len(conflicts) > 0,
                'conflicts': conflicts
            }
        except Exception as e:
            self.logger.error(f"Allergy conflict check failed: {e}")
            return {'has_conflicts': None, 'error': str(e)}

    def _check_duplicate_therapy(
        self,
        medications: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Check for duplicate or redundant therapy.

        Args:
            medications: List of medication dicts

        Returns:
            List of potential duplicates
        """
        duplicates = []
        drug_names = [med.get('medication_name', '').lower() for med in medications]

        # Check for exact duplicates
        seen = set()
        for name in drug_names:
            if name in seen:
                duplicates.append(f"Duplicate prescription: {name}")
            seen.add(name)

        # Check RxNorm data for same drug class
        drug_classes = {}
        for med in medications:
            drug_name = med.get('medication_name', '')
            rxnorm_code = med.get('rxnorm_code', '')

            if rxnorm_code and rxnorm_code in self.rxnorm_data:
                drug_class = self.rxnorm_data[rxnorm_code].get('class', '')
                if drug_class:
                    if drug_class in drug_classes:
                        duplicates.append(
                            f"Multiple drugs in same class ({drug_class}): "
                            f"{drug_classes[drug_class]} and {drug_name}"
                        )
                    else:
                        drug_classes[drug_class] = drug_name

        return duplicates

    def _is_controlled_substance(self, medication: Dict[str, Any]) -> bool:
        """
        Check if medication is a controlled substance.

        Args:
            medication: Medication dict

        Returns:
            True if controlled substance
        """
        rxnorm_code = medication.get('rxnorm_code', '')

        if rxnorm_code and rxnorm_code in self.rxnorm_data:
            schedule = self.rxnorm_data[rxnorm_code].get('schedule', '')
            return schedule in self.controlled_schedules

        # Fallback: check drug name for common controlled substances
        drug_name = medication.get('medication_name', '').lower()
        controlled_keywords = [
            'oxycodone', 'hydrocodone', 'morphine', 'fentanyl',
            'alprazolam', 'diazepam', 'lorazepam', 'clonazepam',
            'adderall', 'ritalin', 'methylphenidate', 'amphetamine'
        ]

        return any(keyword in drug_name for keyword in controlled_keywords)

    def _check_controlled_substance_compliance(
        self,
        medication: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """
        Check controlled substance prescription compliance.

        Args:
            medication: Medication dict
            context: Processing context

        Returns:
            Dict with compliance check results
        """
        warnings = []
        prescriber = context.metadata.get('prescriber', {})

        # Check for DEA number
        if not prescriber.get('dea'):
            warnings.append("Controlled substance prescribed without DEA number")

        # Check quantity
        quantity = medication.get('quantity')
        if quantity and isinstance(quantity, str):
            try:
                quantity = int(quantity)
            except:
                quantity = None

        if quantity and quantity > 90:
            warnings.append(
                f"Controlled substance quantity ({quantity}) exceeds typical 90-day supply"
            )

        # Check refills
        refills = medication.get('refills')
        if refills and isinstance(refills, str):
            try:
                refills = int(refills)
            except:
                refills = None

        rxnorm_code = medication.get('rxnorm_code', '')
        schedule = ''
        if rxnorm_code and rxnorm_code in self.rxnorm_data:
            schedule = self.rxnorm_data[rxnorm_code].get('schedule', '')

        if schedule == 'Schedule II' and refills and refills > 0:
            warnings.append("Schedule II controlled substance cannot have refills")
        elif schedule in ['Schedule III', 'Schedule IV'] and refills and refills > 5:
            warnings.append(
                f"{schedule} controlled substance cannot exceed 5 refills"
            )

        return {
            'compliant': len(warnings) == 0,
            'warnings': warnings
        }
