# ============================================================================
# src/medical_ingestion/core/prompt_manager.py
# ============================================================================
"""
Prompt Manager (Prompt Studio Pattern)

Inspired by Unstract's Prompt Studio. Provides:
- Configurable extraction prompts stored in YAML/JSON files
- Prompt templates with variable substitution
- Prompt versioning and A/B testing support
- Domain-specific prompt customization
- Prompt chaining for complex extractions

Key insight from Unstract: Prompts are the core of extraction accuracy.
Making them configurable and optimizable is critical.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


# Default prompt directory
DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class PromptTemplate:
    """
    A configurable prompt template.

    Supports variable substitution with {variable_name} syntax.
    """
    name: str
    template: str
    description: str = ""
    category: str = "general"  # "extraction", "classification", "validation"
    version: str = "1.0"
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def render(self, **kwargs) -> str:
        """
        Render the template with variable substitution.

        Args:
            **kwargs: Variables to substitute

        Returns:
            Rendered prompt string
        """
        result = self.template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            result = result.replace(placeholder, str(value))
        return result

    def get_variables(self) -> List[str]:
        """Extract variable names from template."""
        return re.findall(r'\{(\w+)\}', self.template)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        return cls(**data)


@dataclass
class PromptChain:
    """
    A chain of prompts executed in sequence.

    Used for complex extractions that require multiple steps.
    """
    name: str
    prompts: List[str]  # List of prompt names
    description: str = ""
    combine_strategy: str = "sequential"  # "sequential", "parallel", "conditional"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Default Medical Extraction Prompts
# =============================================================================
# These are the built-in prompts - users can override with custom prompts

DEFAULT_PROMPTS = {
    # Patient Demographics
    "extract_patient": PromptTemplate(
        name="extract_patient",
        template="""Extract patient demographic information from this medical document.

Document text:
{text}

Extract these fields if present:
- name: Patient's full name
- dob: Date of birth (any format)
- gender: Male/Female/M/F
- mrn: Medical record number or patient ID
- address: Full address
- phone: Phone number
- insurance: Insurance information

Return as JSON object. Use null for fields not found.

JSON:""",
        description="Extract patient demographics from any medical document",
        category="extraction",
        variables=["text"]
    ),

    # Test Results (Labs, Vitals, Measurements)
    "extract_test_results": PromptTemplate(
        name="extract_test_results",
        template="""Extract ALL test results, measurements, and values from this medical document.

Document text:
{text}

For EACH test, measurement, or value found, extract:
- name: The name of the test or measurement (e.g., "Hemoglobin", "Blood Pressure", "Glucose")
- value: The numeric or text value
- unit: Units of measurement if present (e.g., "mg/dL", "mmHg", "g/dL")
- reference_range: Normal range if present (e.g., "10-20", "< 100")
- abnormal_flag: Flag if present (H, L, HIGH, LOW, CRITICAL, ABNORMAL, or null)
- category: Type of test ("lab", "vital", "imaging_measurement", or "other")

Return JSON array of objects. Include ALL quantitative values you can find - lab results, vital signs, imaging measurements, anything with a number.

JSON:""",
        description="Extract all test results and measurements",
        category="extraction",
        variables=["text"]
    ),

    # Medications
    "extract_medications": PromptTemplate(
        name="extract_medications",
        template="""Extract ALL medications mentioned in this medical document.

Document text:
{text}

For EACH medication found, extract:
- name: Drug name as written (e.g., "Lisinopril", "Metformin")
- strength: Dosage strength if present (e.g., "10mg", "500mg")
- route: How taken if mentioned (oral, topical, IV, IM, subcutaneous, etc.)
- frequency: How often if mentioned (daily, BID, TID, PRN, etc.)
- quantity: Amount dispensed if present
- refills: Number of refills if present
- instructions: Any directions or sig codes
- prescriber: Prescriber name if present
- status: current, discontinued, new, or null if unknown

Return JSON array of objects. Include current medications, new prescriptions, discontinued meds, medication allergies if mentioned as drugs.

JSON:""",
        description="Extract all medications from any medical document",
        category="extraction",
        variables=["text"]
    ),

    # Clinical Findings
    "extract_findings": PromptTemplate(
        name="extract_findings",
        template="""Extract ALL clinical findings, impressions, and conclusions from this medical document.

Document text:
{text}

For EACH finding, extract:
- finding: The clinical finding, impression, or diagnosis
- category: One of "diagnosis", "impression", "recommendation", "assessment", "history"
- severity: If indicated ("normal", "abnormal", "critical", "mild", "moderate", "severe") or null
- location: Body part or system if relevant (e.g., "chest", "liver", "cardiovascular")

Return JSON array of objects. Include all diagnoses, impressions, assessments, recommendations, and significant clinical observations.

JSON:""",
        description="Extract clinical findings and impressions",
        category="extraction",
        variables=["text"]
    ),

    # Dates and Providers
    "extract_dates_providers": PromptTemplate(
        name="extract_dates_providers",
        template="""Extract dates and healthcare provider information from this medical document.

Document text:
{text}

Extract:
1. dates: Array of date objects with:
   - date_type: "collection", "report", "service", "birth", "admission", "discharge", or "other"
   - date_value: The date in original format

2. providers: Array of provider objects with:
   - name: Provider's name
   - role: "physician", "nurse", "technician", "specialist", etc.
   - specialty: If mentioned (e.g., "cardiology", "radiology")

3. organizations: Array of organization objects with:
   - name: Facility/lab/hospital name
   - address: Address if present
   - phone: Phone if present

Return JSON object with "dates", "providers", and "organizations" arrays.

JSON:""",
        description="Extract dates, providers, and organizations",
        category="extraction",
        variables=["text"]
    ),

    # Document Classification
    "classify_document": PromptTemplate(
        name="classify_document",
        template="""Classify this medical document into one of the following categories:
- lab_report: Laboratory test results
- prescription: Medication prescriptions
- radiology_report: X-ray, CT, MRI reports
- pathology_report: Biopsy, tissue analysis
- clinical_note: Progress notes, encounter notes
- discharge_summary: Hospital discharge documents
- referral: Referral letters
- insurance: Insurance documents, EOBs
- unknown: Cannot determine type

Document text:
{text}

Respond with JSON containing:
- document_type: The category from above
- confidence: 0.0-1.0 confidence score
- reasoning: Brief explanation

JSON:""",
        description="Classify medical document type",
        category="classification",
        variables=["text"]
    ),

    # Guided Extraction (with hints from similar documents)
    "extract_with_hints": PromptTemplate(
        name="extract_with_hints",
        template="""Extract medical information from this document.

{hints}

Document text:
{text}

Extract ALL relevant information following the patterns shown in the hints above.
Return a JSON object with the extracted fields.

JSON:""",
        description="Extract with hints from similar documents",
        category="extraction",
        variables=["text", "hints"]
    ),

    # Validation prompt
    "validate_extraction": PromptTemplate(
        name="validate_extraction",
        template="""Review this extraction for accuracy and completeness.

Original document:
{text}

Extracted data:
{extraction}

Check for:
1. Missing important values
2. Incorrectly extracted values
3. Values that don't match the document
4. Formatting issues

Return JSON with:
- is_valid: true/false
- confidence: 0.0-1.0
- issues: Array of issues found
- corrections: Object with field corrections if any

JSON:""",
        description="Validate extraction accuracy",
        category="validation",
        variables=["text", "extraction"]
    )
}


class PromptManager:
    """
    Manages extraction prompts with configurability.

    Features (Unstract-inspired):
    - Load prompts from YAML/JSON files
    - Override defaults with custom prompts
    - Prompt versioning
    - Variable substitution
    - Prompt chains for complex extractions

    Usage:
        manager = PromptManager()
        prompt = manager.get_prompt("extract_test_results")
        rendered = prompt.render(text=document_text)
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        self.prompts_dir = prompts_dir or Path(self.config.get('prompts_dir', DEFAULT_PROMPTS_DIR))
        self.prompts_dir = Path(self.prompts_dir)

        # Initialize with default prompts
        self._prompts: Dict[str, PromptTemplate] = {}
        self._chains: Dict[str, PromptChain] = {}
        self._load_defaults()

        # Load custom prompts from directory
        if self.prompts_dir.exists():
            self._load_from_directory()

    def _load_defaults(self):
        """Load default built-in prompts."""
        for name, template in DEFAULT_PROMPTS.items():
            self._prompts[name] = template
        logger.debug(f"Loaded {len(self._prompts)} default prompts")

    def _load_from_directory(self):
        """Load custom prompts from prompts directory."""
        if not self.prompts_dir.exists():
            return

        # Load YAML files
        if HAS_YAML:
            for yaml_file in self.prompts_dir.glob("*.yaml"):
                self._load_yaml_file(yaml_file)
            for yaml_file in self.prompts_dir.glob("*.yml"):
                self._load_yaml_file(yaml_file)

        # Load JSON files
        for json_file in self.prompts_dir.glob("*.json"):
            self._load_json_file(json_file)

    def _load_yaml_file(self, path: Path):
        """Load prompts from YAML file."""
        if not HAS_YAML:
            logger.warning(f"PyYAML not installed, cannot load {path}")
            return

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            if not data:
                return

            # Handle prompts
            prompts = data.get('prompts', [])
            for prompt_data in prompts:
                if isinstance(prompt_data, dict) and 'name' in prompt_data:
                    template = PromptTemplate.from_dict(prompt_data)
                    self._prompts[template.name] = template
                    logger.info(f"Loaded custom prompt: {template.name}")

            # Handle chains
            chains = data.get('chains', [])
            for chain_data in chains:
                if isinstance(chain_data, dict) and 'name' in chain_data:
                    chain = PromptChain(**chain_data)
                    self._chains[chain.name] = chain
                    logger.info(f"Loaded prompt chain: {chain.name}")

        except Exception as e:
            logger.error(f"Error loading {path}: {e}")

    def _load_json_file(self, path: Path):
        """Load prompts from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)

            if not data:
                return

            # Handle prompts
            prompts = data.get('prompts', [])
            for prompt_data in prompts:
                if isinstance(prompt_data, dict) and 'name' in prompt_data:
                    template = PromptTemplate.from_dict(prompt_data)
                    self._prompts[template.name] = template
                    logger.info(f"Loaded custom prompt: {template.name}")

            # Handle chains
            chains = data.get('chains', [])
            for chain_data in chains:
                if isinstance(chain_data, dict) and 'name' in chain_data:
                    chain = PromptChain(**chain_data)
                    self._chains[chain.name] = chain

        except Exception as e:
            logger.error(f"Error loading {path}: {e}")

    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.

        Args:
            name: Prompt name

        Returns:
            PromptTemplate or None if not found
        """
        return self._prompts.get(name)

    def render_prompt(self, name: str, **kwargs) -> Optional[str]:
        """
        Get and render a prompt in one step.

        Args:
            name: Prompt name
            **kwargs: Variables for substitution

        Returns:
            Rendered prompt string or None
        """
        template = self.get_prompt(name)
        if template:
            return template.render(**kwargs)
        return None

    def get_chain(self, name: str) -> Optional[PromptChain]:
        """Get a prompt chain by name."""
        return self._chains.get(name)

    def list_prompts(self, category: str = None) -> List[str]:
        """
        List available prompt names.

        Args:
            category: Optional category filter

        Returns:
            List of prompt names
        """
        if category:
            return [
                name for name, prompt in self._prompts.items()
                if prompt.category == category
            ]
        return list(self._prompts.keys())

    def add_prompt(self, template: PromptTemplate):
        """
        Add or update a prompt template.

        Args:
            template: PromptTemplate to add
        """
        self._prompts[template.name] = template
        logger.info(f"Added prompt: {template.name}")

    def add_chain(self, chain: PromptChain):
        """
        Add or update a prompt chain.

        Args:
            chain: PromptChain to add
        """
        self._chains[chain.name] = chain
        logger.info(f"Added chain: {chain.name}")

    def save_prompts(self, path: Path, format: str = "yaml"):
        """
        Save current prompts to file.

        Args:
            path: Output file path
            format: "yaml" or "json"
        """
        data = {
            "prompts": [p.to_dict() for p in self._prompts.values()],
            "chains": [c.to_dict() for c in self._chains.values()]
        }

        if format == "yaml" and HAS_YAML:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._prompts)} prompts to {path}")

    def get_extraction_prompts(self) -> Dict[str, PromptTemplate]:
        """Get all extraction prompts."""
        return {
            name: prompt for name, prompt in self._prompts.items()
            if prompt.category == "extraction"
        }

    def create_guided_prompt(
        self,
        base_prompt_name: str,
        hints: Dict[str, Any]
    ) -> str:
        """
        Create a prompt with extraction hints from similar documents.

        This is key to Unstract's approach: use hints from successful
        extractions to guide current extraction.

        Args:
            base_prompt_name: Base prompt to use
            hints: Hints from similar documents

        Returns:
            Rendered prompt with hints
        """
        base_prompt = self.get_prompt(base_prompt_name)
        if not base_prompt:
            return ""

        # Format hints
        hints_text = ""
        if hints.get('field_examples'):
            hints_text = "[Extraction hints from similar documents]\n"
            for field_name, examples in hints['field_examples'].items():
                if examples:
                    example_values = [str(e.get('value', '')) for e in examples[:2]]
                    hints_text += f"- {field_name}: Examples: {', '.join(example_values)}\n"
            hints_text += "\n"

        # Use the guided extraction prompt if we have hints
        if hints_text:
            guided_prompt = self.get_prompt("extract_with_hints")
            if guided_prompt:
                return guided_prompt.template.replace("{hints}", hints_text)

        return base_prompt.template


# Singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(config: Dict[str, Any] = None) -> PromptManager:
    """Get shared PromptManager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager(config=config)
    return _prompt_manager
