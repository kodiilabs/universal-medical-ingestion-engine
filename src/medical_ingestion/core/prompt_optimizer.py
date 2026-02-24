# ============================================================================
# src/medical_ingestion/core/prompt_optimizer.py
# ============================================================================
"""
Prompt Optimizer for Systematic LLM Accuracy Improvement

Inspired by Unstract's Prompt Studio:
- Test prompts against ground truth documents
- Track field-level accuracy metrics
- Compare multiple prompt variants
- Identify common error patterns

Usage:
    optimizer = PromptOptimizer(config)
    results = await optimizer.evaluate_prompt(prompt, test_docs)
    print(f"Accuracy: {results.overall_accuracy:.1%}")
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import logging
import json
import asyncio
from datetime import datetime


@dataclass
class FieldAccuracy:
    """Accuracy metrics for a single field."""
    field_name: str
    total_expected: int = 0
    total_extracted: int = 0
    correct: int = 0
    incorrect: int = 0
    missing: int = 0
    extra: int = 0

    @property
    def precision(self) -> float:
        if self.total_extracted == 0:
            return 0.0
        return self.correct / self.total_extracted

    @property
    def recall(self) -> float:
        if self.total_expected == 0:
            return 0.0
        return self.correct / self.total_expected

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class DocumentResult:
    """Evaluation result for a single document."""
    doc_id: str
    doc_path: str
    fields_expected: int = 0
    fields_extracted: int = 0
    fields_correct: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.fields_expected == 0:
            return 0.0
        return self.fields_correct / self.fields_expected


@dataclass
class PromptEvaluation:
    """Complete evaluation result for a prompt."""
    prompt_id: str
    prompt_text: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Overall metrics
    documents_tested: int = 0
    overall_accuracy: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0

    # Field-level metrics
    field_accuracies: Dict[str, FieldAccuracy] = field(default_factory=dict)

    # Document-level results
    document_results: List[DocumentResult] = field(default_factory=list)

    # Error analysis
    common_errors: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    total_time: float = 0.0
    avg_time_per_doc: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt_id': self.prompt_id,
            'timestamp': self.timestamp.isoformat(),
            'documents_tested': self.documents_tested,
            'overall_accuracy': self.overall_accuracy,
            'overall_precision': self.overall_precision,
            'overall_recall': self.overall_recall,
            'overall_f1': self.overall_f1,
            'field_accuracies': {
                name: {
                    'precision': fa.precision,
                    'recall': fa.recall,
                    'f1': fa.f1,
                    'correct': fa.correct,
                    'incorrect': fa.incorrect,
                    'missing': fa.missing
                }
                for name, fa in self.field_accuracies.items()
            },
            'common_errors': self.common_errors[:10],
            'total_time': self.total_time,
            'avg_time_per_doc': self.avg_time_per_doc
        }


@dataclass
class GroundTruthDocument:
    """Document with labeled ground truth values."""
    doc_id: str
    pdf_path: Path
    text: str
    ground_truth: Dict[str, Any]  # field_name -> expected_value
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptOptimizer:
    """
    Systematic prompt testing and optimization.

    Tests prompts against labeled documents and tracks accuracy metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._llm_client = None

        # Results storage
        results_dir = Path(config.get('data_dir', 'data')) / 'prompt_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir

    @property
    def llm_client(self):
        if self._llm_client is None:
            from ..medgemma.client import create_client
            self._llm_client = create_client(self.config)
        return self._llm_client

    def load_ground_truth(
        self,
        ground_truth_dir: Path
    ) -> List[GroundTruthDocument]:
        """
        Load ground truth documents from directory.

        Expected structure:
            ground_truth_dir/
                doc1.pdf
                doc1.json  # {"hemoglobin": 12.5, "wbc": 7200, ...}
                doc2.pdf
                doc2.json
        """
        docs = []

        for json_path in ground_truth_dir.glob('*.json'):
            pdf_path = json_path.with_suffix('.pdf')
            if not pdf_path.exists():
                self.logger.warning(f"No PDF for {json_path}, skipping")
                continue

            try:
                with open(json_path) as f:
                    ground_truth = json.load(f)

                # Extract text
                from ..extractors.text_extractor import TextExtractor
                extractor = TextExtractor()
                text = extractor.extract_text(pdf_path)

                doc = GroundTruthDocument(
                    doc_id=json_path.stem,
                    pdf_path=pdf_path,
                    text=text,
                    ground_truth=ground_truth.get('values', ground_truth),
                    metadata=ground_truth.get('metadata', {})
                )
                docs.append(doc)

            except Exception as e:
                self.logger.error(f"Failed to load {json_path}: {e}")

        self.logger.info(f"Loaded {len(docs)} ground truth documents")
        return docs

    async def evaluate_prompt(
        self,
        prompt_template: str,
        test_docs: List[GroundTruthDocument],
        prompt_id: Optional[str] = None,
        value_tolerance: float = 0.01
    ) -> PromptEvaluation:
        """
        Evaluate a prompt against ground truth documents.

        Args:
            prompt_template: Prompt with {text} placeholder
            test_docs: List of ground truth documents
            prompt_id: Identifier for this prompt
            value_tolerance: Numeric tolerance for value matching

        Returns:
            PromptEvaluation with accuracy metrics
        """
        import time
        start_time = time.time()

        if prompt_id is None:
            import hashlib
            prompt_id = hashlib.md5(prompt_template.encode()).hexdigest()[:8]

        evaluation = PromptEvaluation(
            prompt_id=prompt_id,
            prompt_text=prompt_template
        )

        # Track field-level metrics
        field_metrics: Dict[str, FieldAccuracy] = {}

        # Error patterns
        error_patterns: List[Dict[str, Any]] = []

        for doc in test_docs:
            doc_result = await self._evaluate_document(
                prompt_template,
                doc,
                value_tolerance
            )

            evaluation.document_results.append(doc_result)

            # Aggregate field metrics
            for error in doc_result.errors:
                field = error['field']
                if field not in field_metrics:
                    field_metrics[field] = FieldAccuracy(field_name=field)

                fa = field_metrics[field]

                if error['type'] == 'correct':
                    fa.correct += 1
                    fa.total_expected += 1
                    fa.total_extracted += 1
                elif error['type'] == 'incorrect':
                    fa.incorrect += 1
                    fa.total_expected += 1
                    fa.total_extracted += 1
                    error_patterns.append(error)
                elif error['type'] == 'missing':
                    fa.missing += 1
                    fa.total_expected += 1
                    error_patterns.append(error)
                elif error['type'] == 'extra':
                    fa.extra += 1
                    fa.total_extracted += 1

        # Calculate overall metrics
        evaluation.documents_tested = len(test_docs)
        evaluation.field_accuracies = field_metrics

        if evaluation.document_results:
            accuracies = [r.accuracy for r in evaluation.document_results]
            evaluation.overall_accuracy = sum(accuracies) / len(accuracies)

            # Overall precision/recall
            total_correct = sum(fa.correct for fa in field_metrics.values())
            total_extracted = sum(fa.total_extracted for fa in field_metrics.values())
            total_expected = sum(fa.total_expected for fa in field_metrics.values())

            if total_extracted > 0:
                evaluation.overall_precision = total_correct / total_extracted
            if total_expected > 0:
                evaluation.overall_recall = total_correct / total_expected
            if evaluation.overall_precision + evaluation.overall_recall > 0:
                evaluation.overall_f1 = 2 * (
                    evaluation.overall_precision * evaluation.overall_recall
                ) / (evaluation.overall_precision + evaluation.overall_recall)

        # Analyze common errors
        evaluation.common_errors = self._analyze_error_patterns(error_patterns)

        # Timing
        evaluation.total_time = time.time() - start_time
        if test_docs:
            evaluation.avg_time_per_doc = evaluation.total_time / len(test_docs)

        # Save results
        self._save_evaluation(evaluation)

        return evaluation

    async def _evaluate_document(
        self,
        prompt_template: str,
        doc: GroundTruthDocument,
        tolerance: float
    ) -> DocumentResult:
        """Evaluate prompt on a single document."""
        result = DocumentResult(
            doc_id=doc.doc_id,
            doc_path=str(doc.pdf_path),
            fields_expected=len(doc.ground_truth)
        )

        try:
            # Format prompt with document text
            prompt = prompt_template.format(text=doc.text[:3000])

            # Run extraction
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.1,
                json_mode=True
            )

            # Parse response
            extracted = self._parse_extraction_response(response.get('text', ''))
            result.fields_extracted = len(extracted)

            # Compare to ground truth
            for field, expected in doc.ground_truth.items():
                field_lower = field.lower().replace(' ', '_')

                if field_lower in extracted:
                    actual = extracted[field_lower]
                    is_correct = self._values_match(expected, actual, tolerance)

                    if is_correct:
                        result.fields_correct += 1
                        result.errors.append({
                            'type': 'correct',
                            'field': field,
                            'expected': expected,
                            'actual': actual
                        })
                    else:
                        result.errors.append({
                            'type': 'incorrect',
                            'field': field,
                            'expected': expected,
                            'actual': actual,
                            'doc_id': doc.doc_id
                        })
                else:
                    result.errors.append({
                        'type': 'missing',
                        'field': field,
                        'expected': expected,
                        'doc_id': doc.doc_id
                    })

            # Check for extra fields
            for field, value in extracted.items():
                field_normalized = field.lower().replace('_', ' ')
                gt_fields = [f.lower().replace('_', ' ') for f in doc.ground_truth.keys()]
                if field_normalized not in gt_fields:
                    result.errors.append({
                        'type': 'extra',
                        'field': field,
                        'actual': value
                    })

        except Exception as e:
            self.logger.error(f"Evaluation failed for {doc.doc_id}: {e}")
            result.errors.append({
                'type': 'error',
                'message': str(e)
            })

        return result

    def _parse_extraction_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into extracted values."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            from json_repair import repair_json
            try:
                data = repair_json(response_text, return_objects=True)
            except Exception:
                return {}

        # Handle common response formats
        if isinstance(data, dict):
            # Check for results array
            if 'results' in data:
                values = {}
                for item in data['results']:
                    name = item.get('test_name', item.get('name', ''))
                    name = name.lower().replace(' ', '_')
                    if name:
                        values[name] = item.get('value')
                return values

            # Direct field mapping
            return {k.lower().replace(' ', '_'): v for k, v in data.items()}

        return {}

    def _values_match(
        self,
        expected: Any,
        actual: Any,
        tolerance: float
    ) -> bool:
        """Check if extracted value matches expected."""
        if expected is None and actual is None:
            return True
        if expected is None or actual is None:
            return False

        # Numeric comparison with tolerance
        try:
            exp_num = float(expected)
            act_num = float(actual) if not isinstance(actual, (int, float)) else actual

            if exp_num == 0:
                return abs(act_num) <= tolerance
            return abs(exp_num - act_num) / abs(exp_num) <= tolerance

        except (ValueError, TypeError):
            pass

        # String comparison (case insensitive)
        return str(expected).lower().strip() == str(actual).lower().strip()

    def _analyze_error_patterns(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze common error patterns."""
        from collections import Counter

        # Group by field
        field_errors = {}
        for error in errors:
            if error['type'] not in ['incorrect', 'missing']:
                continue

            field = error['field']
            if field not in field_errors:
                field_errors[field] = []
            field_errors[field].append(error)

        # Find most common error fields
        error_counts = Counter(
            error['field'] for error in errors
            if error['type'] in ['incorrect', 'missing']
        )

        patterns = []
        for field, count in error_counts.most_common(10):
            field_errs = field_errors.get(field, [])

            # Analyze error types
            incorrect = [e for e in field_errs if e['type'] == 'incorrect']
            missing = [e for e in field_errs if e['type'] == 'missing']

            pattern = {
                'field': field,
                'total_errors': count,
                'incorrect_count': len(incorrect),
                'missing_count': len(missing),
            }

            # Sample errors
            if incorrect:
                pattern['sample_incorrect'] = {
                    'expected': incorrect[0]['expected'],
                    'actual': incorrect[0]['actual']
                }

            patterns.append(pattern)

        return patterns

    def _save_evaluation(self, evaluation: PromptEvaluation):
        """Save evaluation results to file."""
        filename = f"{evaluation.prompt_id}_{evaluation.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(evaluation.to_dict(), f, indent=2)

        self.logger.info(f"Saved evaluation to {filepath}")

    async def compare_prompts(
        self,
        prompts: List[Dict[str, str]],  # [{"id": "v1", "prompt": "..."}]
        test_docs: List[GroundTruthDocument]
    ) -> Dict[str, PromptEvaluation]:
        """
        Compare multiple prompt variants.

        Returns dict of prompt_id -> PromptEvaluation sorted by accuracy.
        """
        results = {}

        for prompt_info in prompts:
            prompt_id = prompt_info['id']
            prompt_text = prompt_info['prompt']

            self.logger.info(f"Evaluating prompt: {prompt_id}")
            evaluation = await self.evaluate_prompt(
                prompt_template=prompt_text,
                test_docs=test_docs,
                prompt_id=prompt_id
            )
            results[prompt_id] = evaluation

        # Log comparison
        self.logger.info("\n=== Prompt Comparison ===")
        for pid, eval in sorted(
            results.items(),
            key=lambda x: x[1].overall_accuracy,
            reverse=True
        ):
            self.logger.info(
                f"  {pid}: {eval.overall_accuracy:.1%} accuracy, "
                f"{eval.overall_f1:.2f} F1"
            )

        return results

    def generate_improvement_suggestions(
        self,
        evaluation: PromptEvaluation
    ) -> List[str]:
        """Generate suggestions for improving the prompt based on errors."""
        suggestions = []

        # Analyze common error patterns
        for pattern in evaluation.common_errors[:5]:
            field = pattern['field']

            if pattern['missing_count'] > pattern['incorrect_count']:
                suggestions.append(
                    f"Add explicit instruction to extract '{field}' - "
                    f"frequently missed ({pattern['missing_count']} times)"
                )
            else:
                sample = pattern.get('sample_incorrect', {})
                if sample:
                    suggestions.append(
                        f"Field '{field}' often extracted incorrectly. "
                        f"Example: expected '{sample.get('expected')}' "
                        f"but got '{sample.get('actual')}'"
                    )

        # Overall accuracy suggestions
        if evaluation.overall_accuracy < 0.80:
            suggestions.append(
                "Consider adding few-shot examples to the prompt"
            )
        if evaluation.overall_precision < evaluation.overall_recall:
            suggestions.append(
                "Too many false positives - add stricter extraction criteria"
            )
        if evaluation.overall_recall < evaluation.overall_precision:
            suggestions.append(
                "Missing values - ensure prompt asks for ALL fields explicitly"
            )

        return suggestions


# Default extraction prompt template
DEFAULT_LAB_EXTRACTION_PROMPT = """Extract all lab test results from this medical report.

Report text:
{text}

Return a JSON object with this structure:
{{
    "results": [
        {{"test_name": "...", "value": ..., "unit": "...", "reference_range": "..."}},
        ...
    ]
}}

Important instructions:
1. Extract EVERY lab value you can find
2. For numeric values, return just the number (e.g., 12.5 not "12.5 g/dL")
3. Normalize test names to lowercase with underscores (e.g., "white_blood_count")
4. Include units separately from values
5. If a value has a flag (H, L, HH, LL), include it in a "flag" field

JSON:"""
