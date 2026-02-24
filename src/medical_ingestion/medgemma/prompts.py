# ============================================================================
# src/medgemma/prompts.py
# ============================================================================
"""
MedGemma Prompt Templates

Provides:
- Medical reasoning prompt templates
- Structured prompts for different tasks
- Prompt formatting utilities
- Few-shot examples
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass


class PromptTask(Enum):
    """Medical reasoning tasks"""
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    RISK_ASSESSMENT = "risk_assessment"
    LAB_INTERPRETATION = "lab_interpretation"
    MEDICATION_REVIEW = "medication_review"
    CLINICAL_SUMMARY = "clinical_summary"
    INFORMATION_EXTRACTION = "information_extraction"
    CLASSIFICATION = "classification"


@dataclass
class PromptTemplate:
    """Prompt template"""
    name: str
    task: PromptTask
    template: str
    description: str
    required_fields: List[str]
    optional_fields: List[str] = None
    examples: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.optional_fields is None:
            self.optional_fields = []
        if self.examples is None:
            self.examples = []

    def format(self, **kwargs) -> str:
        """
        Format template with provided values.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        # Check required fields
        missing = [f for f in self.required_fields if f not in kwargs]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return self.template.format(**kwargs)


class MedGemmaPrompts:
    """
    Collection of MedGemma prompt templates.
    """

    # Diagnosis prompts
    DIAGNOSIS_TEMPLATE = PromptTemplate(
        name="diagnosis",
        task=PromptTask.DIAGNOSIS,
        template="""You are a medical AI assistant. Based on the patient information provided, suggest possible diagnoses.

Patient Information:
{patient_info}

Chief Complaint: {chief_complaint}

{additional_context}

Provide a differential diagnosis with the most likely conditions ranked by probability. For each diagnosis, explain your reasoning based on the symptoms and patient history.

Response format:
1. [Diagnosis name] - [Probability: High/Medium/Low]
   Reasoning: [Explanation]

2. [Diagnosis name] - [Probability: High/Medium/Low]
   Reasoning: [Explanation]
""",
        description="Generate differential diagnosis from patient data",
        required_fields=["patient_info", "chief_complaint"],
        optional_fields=["additional_context"],
    )

    # Lab interpretation prompt
    LAB_INTERPRETATION_TEMPLATE = PromptTemplate(
        name="lab_interpretation",
        task=PromptTask.LAB_INTERPRETATION,
        template="""You are a medical AI assistant specializing in laboratory result interpretation.

Patient Context:
{patient_context}

Laboratory Results:
{lab_results}

Analyze these laboratory results and provide:
1. Clinical interpretation of abnormal values
2. Possible medical conditions indicated by the results
3. Recommended follow-up tests or actions

Focus on clinically significant findings and their implications.

Response format (JSON):
{{
  "abnormal_findings": [
    {{
      "test": "test name",
      "value": "result",
      "reference_range": "range",
      "interpretation": "clinical meaning",
      "severity": "mild/moderate/severe"
    }}
  ],
  "clinical_significance": "overall interpretation",
  "differential_diagnoses": ["condition1", "condition2"],
  "recommendations": ["recommendation1", "recommendation2"]
}}
""",
        description="Interpret laboratory test results",
        required_fields=["patient_context", "lab_results"],
    )

    # Treatment recommendation prompt
    TREATMENT_TEMPLATE = PromptTemplate(
        name="treatment",
        task=PromptTask.TREATMENT,
        template="""You are a medical AI assistant providing treatment recommendations.

Patient Information:
{patient_info}

Diagnosis: {diagnosis}

Relevant Medical History:
{medical_history}

Current Medications:
{current_medications}

Based on this information, suggest evidence-based treatment options. Consider:
- First-line treatments
- Alternative options
- Contraindications based on patient history
- Drug interactions with current medications

Response format:
**Recommended Treatment Plan:**

1. **Primary Treatment:**
   - Medication/Intervention: [name]
   - Dosage: [details]
   - Duration: [timeframe]
   - Rationale: [explanation]

2. **Alternative Options:**
   - [List alternatives with brief rationale]

3. **Precautions:**
   - [List any warnings or contraindications]

4. **Monitoring:**
   - [Recommended follow-up and monitoring]
""",
        description="Generate treatment recommendations",
        required_fields=["patient_info", "diagnosis", "medical_history", "current_medications"],
    )

    # Risk assessment prompt
    RISK_ASSESSMENT_TEMPLATE = PromptTemplate(
        name="risk_assessment",
        task=PromptTask.RISK_ASSESSMENT,
        template="""You are a medical AI assistant performing risk assessment.

Patient Data:
{patient_data}

Risk Factors:
{risk_factors}

Assess the patient's risk for: {condition}

Provide a comprehensive risk assessment including:
1. Overall risk level (Low/Moderate/High/Very High)
2. Contributing risk factors
3. Risk score if applicable
4. Prevention recommendations

Response format (JSON):
{{
  "condition": "{condition}",
  "overall_risk": "Low/Moderate/High/Very High",
  "risk_score": "score if applicable",
  "contributing_factors": [
    {{
      "factor": "factor name",
      "impact": "low/moderate/high",
      "modifiable": true/false
    }}
  ],
  "recommendations": [
    "recommendation 1",
    "recommendation 2"
  ]
}}
""",
        description="Assess patient risk for specific conditions",
        required_fields=["patient_data", "risk_factors", "condition"],
    )

    # Medication review prompt
    MEDICATION_REVIEW_TEMPLATE = PromptTemplate(
        name="medication_review",
        task=PromptTask.MEDICATION_REVIEW,
        template="""You are a medical AI assistant performing medication review.

Patient Information:
Age: {age}
Conditions: {conditions}

Current Medications:
{medications}

Review the medication list for:
1. Drug-drug interactions
2. Duplicate therapies
3. Age-inappropriate medications
4. Contraindications based on patient conditions
5. Dosing concerns

Response format (JSON):
{{
  "interactions": [
    {{
      "medications": ["drug1", "drug2"],
      "severity": "mild/moderate/severe",
      "description": "interaction description",
      "recommendation": "action to take"
    }}
  ],
  "concerns": [
    {{
      "medication": "drug name",
      "issue": "description",
      "severity": "low/medium/high",
      "recommendation": "suggested action"
    }}
  ],
  "overall_assessment": "summary of medication safety"
}}
""",
        description="Review medication list for safety issues",
        required_fields=["age", "conditions", "medications"],
    )

    # Clinical summary prompt
    CLINICAL_SUMMARY_TEMPLATE = PromptTemplate(
        name="clinical_summary",
        task=PromptTask.CLINICAL_SUMMARY,
        template="""You are a medical AI assistant creating a clinical summary.

Patient Record:
{patient_record}

Create a concise clinical summary highlighting:
1. Key medical problems
2. Active diagnoses
3. Current treatment plan
4. Recent significant events
5. Pending issues requiring follow-up

Format the summary in a structured, easy-to-read format suitable for physician review.

Response format:
**Clinical Summary**

**Patient:** [Demographics]

**Active Problems:**
1. [Problem 1]
2. [Problem 2]

**Current Medications:**
- [Medication list]

**Recent Events:**
- [Significant events]

**Pending/Follow-up:**
- [Items requiring attention]

**Assessment:**
[Brief overall assessment]
""",
        description="Generate clinical summary from patient records",
        required_fields=["patient_record"],
    )

    # Information extraction prompt
    EXTRACTION_TEMPLATE = PromptTemplate(
        name="extraction",
        task=PromptTask.INFORMATION_EXTRACTION,
        template="""You are a medical AI assistant extracting structured information from clinical text.

Clinical Text:
{clinical_text}

Extract the following information in JSON format:
{fields_to_extract}

Be precise and only include information explicitly stated in the text. Use null for missing values.

Response format (JSON):
{{
  {json_schema}
}}
""",
        description="Extract structured data from clinical text",
        required_fields=["clinical_text", "fields_to_extract", "json_schema"],
    )

    # Classification prompt
    CLASSIFICATION_TEMPLATE = PromptTemplate(
        name="classification",
        task=PromptTask.CLASSIFICATION,
        template="""You are a medical AI assistant performing document classification.

Document Text:
{document_text}

Classify this document into one of the following categories:
{categories}

Provide:
1. The most appropriate category
2. Confidence level (0-100%)
3. Brief reasoning

Response format (JSON):
{{
  "category": "selected category",
  "confidence": 95,
  "reasoning": "explanation for classification"
}}
""",
        description="Classify medical documents",
        required_fields=["document_text", "categories"],
    )

    @classmethod
    def get_template(cls, task: PromptTask) -> PromptTemplate:
        """
        Get template for a specific task.

        Args:
            task: Task type

        Returns:
            PromptTemplate
        """
        template_map = {
            PromptTask.DIAGNOSIS: cls.DIAGNOSIS_TEMPLATE,
            PromptTask.LAB_INTERPRETATION: cls.LAB_INTERPRETATION_TEMPLATE,
            PromptTask.TREATMENT: cls.TREATMENT_TEMPLATE,
            PromptTask.RISK_ASSESSMENT: cls.RISK_ASSESSMENT_TEMPLATE,
            PromptTask.MEDICATION_REVIEW: cls.MEDICATION_REVIEW_TEMPLATE,
            PromptTask.CLINICAL_SUMMARY: cls.CLINICAL_SUMMARY_TEMPLATE,
            PromptTask.INFORMATION_EXTRACTION: cls.EXTRACTION_TEMPLATE,
            PromptTask.CLASSIFICATION: cls.CLASSIFICATION_TEMPLATE,
        }

        return template_map.get(task)

    @classmethod
    def list_templates(cls) -> List[Dict[str, str]]:
        """
        List all available templates.

        Returns:
            List of template information
        """
        templates = []
        for task in PromptTask:
            template = cls.get_template(task)
            if template:
                templates.append({
                    "name": template.name,
                    "task": task.value,
                    "description": template.description,
                    "required_fields": template.required_fields,
                })

        return templates


class PromptBuilder:
    """
    Utility for building complex prompts.
    """

    def __init__(self, task: PromptTask):
        """
        Initialize prompt builder.

        Args:
            task: Task type
        """
        self.task = task
        self.template = MedGemmaPrompts.get_template(task)
        self.values: Dict[str, Any] = {}

    def set(self, field: str, value: Any) -> 'PromptBuilder':
        """
        Set field value.

        Args:
            field: Field name
            value: Field value

        Returns:
            Self for chaining
        """
        self.values[field] = value
        return self

    def set_multiple(self, **kwargs) -> 'PromptBuilder':
        """
        Set multiple field values.

        Args:
            **kwargs: Field values

        Returns:
            Self for chaining
        """
        self.values.update(kwargs)
        return self

    def build(self) -> str:
        """
        Build final prompt.

        Returns:
            Formatted prompt string
        """
        # Fill in optional fields with defaults
        for field in self.template.optional_fields:
            if field not in self.values:
                self.values[field] = ""

        return self.template.format(**self.values)


def create_lab_interpretation_prompt(
    patient_context: str,
    lab_results: str,
) -> str:
    """
    Create lab interpretation prompt.

    Args:
        patient_context: Patient background
        lab_results: Lab test results

    Returns:
        Formatted prompt
    """
    return PromptBuilder(PromptTask.LAB_INTERPRETATION) \
        .set("patient_context", patient_context) \
        .set("lab_results", lab_results) \
        .build()


def create_diagnosis_prompt(
    patient_info: str,
    chief_complaint: str,
    additional_context: str = "",
) -> str:
    """
    Create diagnosis prompt.

    Args:
        patient_info: Patient information
        chief_complaint: Chief complaint
        additional_context: Additional context

    Returns:
        Formatted prompt
    """
    return PromptBuilder(PromptTask.DIAGNOSIS) \
        .set("patient_info", patient_info) \
        .set("chief_complaint", chief_complaint) \
        .set("additional_context", additional_context) \
        .build()


def create_extraction_prompt(
    clinical_text: str,
    fields: List[str],
    schema: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create information extraction prompt.

    Args:
        clinical_text: Clinical text to extract from
        fields: List of fields to extract
        schema: JSON schema for output

    Returns:
        Formatted prompt
    """
    fields_str = "\n".join(f"- {field}" for field in fields)

    if schema is None:
        schema = {field: "value" for field in fields}

    schema_str = ",\n  ".join(f'"{k}": "{v}"' for k, v in schema.items())

    return PromptBuilder(PromptTask.INFORMATION_EXTRACTION) \
        .set("clinical_text", clinical_text) \
        .set("fields_to_extract", fields_str) \
        .set("json_schema", schema_str) \
        .build()
