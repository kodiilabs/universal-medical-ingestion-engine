# ============================================================================
# src/medical_ingestion/core/__init__.py
# ============================================================================
"""
Core components for the medical ingestion engine.
"""

from .context import ProcessingContext, ExtractedValue
from .agent_base import Agent
from .confidence import ConfidenceCalculator
from .audit import AuditLogger

# Consensus extraction (Unstract-inspired)
from .extraction_engine import (
    ExtractionEngine,
    ConsensusResult,
    ConsensusField,
    MethodResult,
    AgreementLevel
)

# Document preprocessing pipeline
from .document_pipeline import (
    DocumentPipeline,
    PipelineResult,
    process_document,
    combine_multipage_results
)

# Extraction-First Pipeline (Unstract-inspired)
from .extraction_first_pipeline import (
    ExtractionFirstPipeline,
    ExtractionFirstResult,
    PipelineStage,
    process_document_extraction_first
)

# Adaptive Retrieval
from .adaptive_retrieval import (
    AdaptiveRetrieval,
    RetrievalStrategy,
    RetrievalResult,
    RetrievalContext,
    retrieve_context
)

from .vector_store import VectorStore, get_vector_store
from .prompt_optimizer import PromptOptimizer, PromptEvaluation

# Prompt Manager (Unstract-inspired Prompt Studio pattern)
from .prompt_manager import (
    PromptManager,
    PromptTemplate,
    PromptChain,
    get_prompt_manager
)
from .config import Config, get_config, get_config_instance, reload_config
from .bbox_utils import (
    validate_bbox,
    normalize_bbox,
    merge_bboxes,
    fix_bbox_ordering,
    bbox_area,
    bbox_overlap,
    find_text_in_word_boxes,
    find_field_value_bbox,
    calculate_bbox_confidence,
    BBoxMatch
)

__all__ = [
    # Core context
    'ProcessingContext',
    'ExtractedValue',
    'Agent',
    'ConfidenceCalculator',
    'AuditLogger',

    # Consensus extraction
    'ExtractionEngine',
    'ConsensusResult',
    'ConsensusField',
    'MethodResult',
    'AgreementLevel',
    'VectorStore',
    'get_vector_store',
    'PromptOptimizer',
    'PromptEvaluation',

    # Document preprocessing pipeline
    'DocumentPipeline',
    'PipelineResult',
    'process_document',
    'combine_multipage_results',

    # Extraction-First Pipeline
    'ExtractionFirstPipeline',
    'ExtractionFirstResult',
    'PipelineStage',
    'process_document_extraction_first',

    # Adaptive Retrieval
    'AdaptiveRetrieval',
    'RetrievalStrategy',
    'RetrievalResult',
    'RetrievalContext',
    'retrieve_context',

    # Configuration
    'Config',
    'get_config',
    'get_config_instance',
    'reload_config',

    # Prompt Manager
    'PromptManager',
    'PromptTemplate',
    'PromptChain',
    'get_prompt_manager',

    # Bounding box utilities
    'validate_bbox',
    'normalize_bbox',
    'merge_bboxes',
    'fix_bbox_ordering',
    'bbox_area',
    'bbox_overlap',
    'find_text_in_word_boxes',
    'find_field_value_bbox',
    'calculate_bbox_confidence',
    'BBoxMatch',
]
