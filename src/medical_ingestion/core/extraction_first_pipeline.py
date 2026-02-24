# ============================================================================
# src/medical_ingestion/core/extraction_first_pipeline.py
# ============================================================================
"""
Extraction-First Processing Pipeline

The main orchestrator for the new Unstract-inspired architecture.

Key features (Unstract-inspired):
- PromptManager for configurable extraction prompts
- AdaptiveRetrieval with vector store for semantic chunk selection
- Similar document retrieval for extraction hints
- Automatic storage of successful extractions for future learning

Flow:
1. UniversalTextExtractor → Clean text + layout from any document
2. Find similar documents → Get extraction hints from vector store
3. ContentAgnosticExtractor (PARALLEL with Classifier) → Generic medical extraction
4. Merge: Classification enriches extraction
5. TypeSpecificEnricher → Add LOINC codes, RxNorm, etc.
6. Store successful extraction → Update vector store for future learning

Key Benefits:
- Extraction works even if classification fails
- Classification uses clean extracted text (better accuracy)
- Classification enriches extraction rather than gating it
- Learns from past extractions (vector store)
- Configurable prompts (PromptManager)
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

if TYPE_CHECKING:
    from .prompt_manager import PromptManager
    from .adaptive_retrieval import AdaptiveRetrieval
    from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Tracking info for a pipeline stage."""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExtractionFirstResult:
    """Complete result from extraction-first pipeline."""

    # Universal extraction (type-agnostic)
    universal_extraction: Any  # GenericMedicalExtraction

    # Classification (optional, for enrichment)
    classification: Optional[Dict[str, Any]] = None

    # Enriched data (type-specific post-processing)
    enriched_extraction: Optional[Any] = None

    # Text extraction result
    text_result: Optional[Any] = None  # UniversalTextResult

    # Processing metadata
    pipeline_stages: List[PipelineStage] = field(default_factory=list)
    total_time: float = 0.0
    confidence: float = 0.0

    # Warnings and review flags
    warnings: List[str] = field(default_factory=list)
    requires_review: bool = False
    review_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "universal_extraction": (
                self.universal_extraction.to_dict()
                if hasattr(self.universal_extraction, 'to_dict')
                else None
            ),
            "classification": self.classification,
            "enriched_extraction": (
                self.enriched_extraction.to_dict()
                if hasattr(self.enriched_extraction, 'to_dict')
                else None
            ),
            "text_result": {
                "full_text": self.text_result.full_text[:1000] if self.text_result else "",
                "page_count": self.text_result.page_count if self.text_result else 0,
                "source_type": self.text_result.source_type.value if self.text_result else None,
                "extraction_method": self.text_result.extraction_method if self.text_result else None,
                "confidence": self.text_result.confidence if self.text_result else 0,
                "consensus_used": 'consensus' in (self.text_result.extraction_method or '').lower() if self.text_result else False,
                "layout": {
                    "has_tables": self.text_result.layout.has_tables if self.text_result else False,
                    "has_sections": self.text_result.layout.has_sections if self.text_result else False,
                    "has_reference_ranges": self.text_result.layout.has_reference_ranges if self.text_result else False,
                } if self.text_result else {}
            },
            "pipeline_stages": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration_seconds": s.duration_seconds,
                    "details": s.details
                }
                for s in self.pipeline_stages
            ],
            "total_time": self.total_time,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "requires_review": self.requires_review,
            "review_reasons": self.review_reasons
        }


class ExtractionFirstPipeline:
    """
    Orchestrates extraction-first processing flow.

    Key features (Unstract-inspired):
    - PromptManager for configurable prompts
    - AdaptiveRetrieval with vector store integration
    - Similar document hints for improved accuracy
    - Automatic learning from successful extractions

    This is the new architecture inspired by Unstract's approach:
    1. Universal text extraction (format-agnostic)
    2. Find similar documents (vector store)
    3. Content-agnostic extraction (parallel with classification)
    4. Type-specific enrichment (post-processing)
    5. Store successful extractions (learning)

    Usage:
        pipeline = ExtractionFirstPipeline()
        result = await pipeline.process(Path("document.pdf"))
        print(result.universal_extraction.test_results)
        print(result.classification)
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        prompt_manager: 'PromptManager' = None,
        retrieval: 'AdaptiveRetrieval' = None,
        vector_store: 'VectorStore' = None
    ):
        self.config = config or {}

        # Lazy-loaded components
        self._text_extractor = None
        self._content_extractor = None
        self._classifier = None
        self._retrieval = retrieval
        self._prompt_manager = prompt_manager
        self._vector_store = vector_store
        self._enrichers = None

        # Configuration
        # Ollama serializes requests by default (OLLAMA_NUM_PARALLEL=1).
        # Only enable parallel extraction for Ollama if user explicitly sets it,
        # or if using a cloud/transformers backend that handles concurrency natively.
        backend = self.config.get('backend', 'ollama')
        default_parallel = backend not in ('ollama', 'local')
        self.parallel_extraction = self.config.get('parallel_extraction', default_parallel)
        self.skip_classification = self.config.get('skip_classification', False)
        self.enrichment_confidence_threshold = self.config.get(
            'enrichment_confidence_threshold', 0.7
        )
        self.default_retrieval_strategy = self.config.get(
            'default_retrieval_strategy', 'router'
        )
        # Vector store settings
        self.store_extractions = self.config.get('store_extractions', True)
        self.min_store_confidence = self.config.get('min_store_confidence', 0.7)

        # Model names for tracking (read from config/env)
        import os
        self.llm_model = self.config.get('ollama_model', os.getenv('OLLAMA_MODEL', 'medgemma-4b-local'))
        self.vlm_model = self.config.get('vlm_model', os.getenv('VLM_MODEL', 'minicpm-v'))
        self.embedding_model = self.config.get('embedding_model', os.getenv('EMBEDDING_MODEL', 'nomic-embed-text-v2-moe:latest'))

        # Consensus extraction setting
        # Default to False - consensus extraction causes issues with local Ollama (timeouts, hallucinations)
        # Enable only for specific use cases (scanned documents with poor OCR results)
        self.use_consensus_extraction = self.config.get('use_consensus_extraction',
            os.getenv('USE_CONSENSUS_EXTRACTION', 'false').lower() in ('true', '1', 'yes'))

        # Cloud extraction setting
        # When USE_CLOUD=True, use Azure GPT-4o vision instead of local VLM/OCR
        self.use_cloud = self.config.get('use_cloud',
            os.getenv('USE_CLOUD', 'false').lower() in ('true', '1', 'yes'))

        # Azure OpenAI extractor (lazy loaded)
        self._azure_extractor = None

    @property
    def prompt_manager(self) -> 'PromptManager':
        """Lazy load prompt manager."""
        if self._prompt_manager is None:
            from .prompt_manager import get_prompt_manager
            self._prompt_manager = get_prompt_manager(self.config)
        return self._prompt_manager

    @property
    def retrieval(self) -> 'AdaptiveRetrieval':
        """Lazy load adaptive retrieval."""
        if self._retrieval is None:
            from .adaptive_retrieval import AdaptiveRetrieval
            self._retrieval = AdaptiveRetrieval(self.config)
        return self._retrieval

    @property
    def vector_store(self) -> Optional['VectorStore']:
        """Lazy load vector store."""
        if self._vector_store is None:
            try:
                from .vector_store import get_vector_store
                self._vector_store = get_vector_store(self.config)
            except Exception as e:
                logger.warning(f"Could not initialize vector store: {e}")
        return self._vector_store

    @property
    def azure_extractor(self):
        """Lazy load Azure OpenAI extractor."""
        if self._azure_extractor is None and self.use_cloud:
            try:
                from ..extractors.azure_openai_client import AzureOpenAIExtractor
                self._azure_extractor = AzureOpenAIExtractor(self.config)
                if not self._azure_extractor.is_configured():
                    logger.warning("Azure OpenAI not configured. Falling back to local extraction.")
                    self._azure_extractor = None
                else:
                    logger.info("Azure OpenAI extractor initialized for cloud extraction")
            except Exception as e:
                logger.warning(f"Could not initialize Azure extractor: {e}")
                self._azure_extractor = None
        return self._azure_extractor

    async def process(
        self,
        document_path: Path,
        classification_hint: Optional[str] = None,
        retrieval_strategy: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> ExtractionFirstResult:
        """
        Process document using extraction-first flow.

        Args:
            document_path: Path to document file
            classification_hint: Optional hint for document type
            retrieval_strategy: Override default retrieval strategy
            progress_callback: Optional callback for real-time progress updates.
                Called with (stage: PipelineStage, all_stages: List[PipelineStage])
                whenever a stage starts or completes.

        Returns:
            ExtractionFirstResult with complete extraction data
        """
        # Overall pipeline timeout: scales with document size.
        # Base: 10 minutes for small docs. Large docs get extra time for OCR.
        # A 19-page scanned PDF needs ~14 min OCR + ~5 min LLM = ~19 min.
        import os
        base_timeout = int(self.config.get(
            'pipeline_timeout',
            os.getenv('PIPELINE_TIMEOUT', '600')
        ))

        # Estimate page count for dynamic timeout
        page_count = 1
        try:
            if document_path.suffix.lower() == '.pdf':
                import fitz
                doc = fitz.open(str(document_path))
                page_count = len(doc)
                doc.close()
        except Exception:
            pass

        # Scale timeout: base + 60s per page (for OCR + LLM extraction per page)
        pipeline_timeout = base_timeout + max(0, (page_count - 3)) * 60
        logger.info(
            f"Pipeline timeout: {pipeline_timeout}s "
            f"(base={base_timeout}s + {page_count} pages)"
        )

        try:
            return await asyncio.wait_for(
                self._process_internal(
                    document_path, classification_hint,
                    retrieval_strategy, progress_callback
                ),
                timeout=pipeline_timeout
            )
        except asyncio.TimeoutError:
            elapsed = time.time()
            logger.error(
                f"Pipeline timeout after {pipeline_timeout}s for {document_path.name}"
            )
            from ..extractors.content_agnostic_extractor import GenericMedicalExtraction
            return ExtractionFirstResult(
                universal_extraction=GenericMedicalExtraction(
                    warnings=[f"Processing timed out after {pipeline_timeout}s"]
                ),
                pipeline_stages=[PipelineStage(
                    name="timeout",
                    status="failed",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    details={"timeout_seconds": pipeline_timeout,
                             "document": document_path.name}
                )],
                total_time=pipeline_timeout,
                confidence=0.0,
                warnings=[f"Processing timed out after {pipeline_timeout // 60} minutes. "
                          "The document may be too large for local processing."],
                requires_review=True,
                review_reasons=["pipeline_timeout"]
            )

    async def _process_internal(
        self,
        document_path: Path,
        classification_hint: Optional[str] = None,
        retrieval_strategy: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> ExtractionFirstResult:
        """Internal processing logic, wrapped by process() with timeout."""
        start_time = time.time()
        stages = []
        warnings = []
        review_reasons = []

        document_path = Path(document_path)

        def notify_progress(stage: PipelineStage):
            """Notify progress callback if set."""
            if progress_callback:
                try:
                    progress_callback(stage, stages)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        # =====================================================================
        # CLOUD PATH: Use Azure GPT-4o Vision for extraction
        # =====================================================================
        # When USE_CLOUD=True, bypass local VLM/OCR and use Azure GPT-4o vision
        # This provides a single API call that handles both OCR and extraction
        if self.use_cloud and self.azure_extractor:
            logger.info("Using Azure GPT-4o cloud extraction (USE_CLOUD=True)")
            return await self._process_with_cloud(
                document_path=document_path,
                classification_hint=classification_hint,
                retrieval_strategy=retrieval_strategy,
                progress_callback=progress_callback
            )

        # =====================================================================
        # LOCAL PATH: Standard extraction with local VLM/OCR
        # =====================================================================

        # =====================================================================
        # Stage 1: Universal Text Extraction
        # =====================================================================
        stage = PipelineStage(name="text_extraction", status="running", started_at=datetime.now())
        stages.append(stage)
        notify_progress(stage)  # Notify: stage started

        def page_progress(current_page, total_pages):
            """Update stage details with per-page progress."""
            stage.details = {
                **(stage.details or {}),
                "current_page": current_page,
                "total_pages": total_pages,
            }
            notify_progress(stage)

        try:
            text_result = await self._extract_text(document_path, page_progress_callback=page_progress)
            stage.status = "completed"
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            # Determine if consensus extraction was used
            is_consensus = 'consensus' in text_result.extraction_method.lower()
            is_vlm = 'vlm' in text_result.extraction_method.lower()
            is_vlm_unified = text_result.extraction_method == 'vlm_unified'
            is_ocr = any(x in text_result.extraction_method.lower() for x in ['ocr', 'paddle', 'tesseract'])

            stage.details = {
                "chars_extracted": len(text_result.full_text),
                "pages": text_result.page_count,
                "source_type": text_result.source_type.value,
                "method": text_result.extraction_method,
                "file_type": document_path.suffix.lower(),
                "file_name": document_path.name,
                "consensus_used": is_consensus,
                "ocr_used": is_ocr,
                "vlm_used": is_vlm,
                "vlm_unified": is_vlm_unified,
                "vlm_model": self.vlm_model if (is_vlm or is_consensus) else None,
                "ocr_engine": None if is_vlm_unified else ("PaddleOCR" if is_ocr else None),
                "description": (
                    f"VLM Unified — {self.vlm_model} (single model, no OCR)" if is_vlm_unified
                    else "Consensus extraction (VLM + OCR in parallel)" if is_consensus
                    else "VLM-based text extraction" if is_vlm
                    else "OCR-based text extraction"
                )
            }

            logger.info(
                f"Text extraction complete: {len(text_result.full_text)} chars, "
                f"{text_result.page_count} pages"
            )
            notify_progress(stage)  # Notify: stage completed

            # Check for extraction warnings
            if text_result.warnings:
                warnings.extend(text_result.warnings)

            if len(text_result.full_text) < 100:
                warnings.append("Very little text extracted - document may be mostly images")
                review_reasons.append("low_text_content")

        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            logger.error(f"Text extraction failed: {e}")
            notify_progress(stage)  # Notify: stage failed

            # Return early with error
            return ExtractionFirstResult(
                universal_extraction=None,
                pipeline_stages=stages,
                total_time=time.time() - start_time,
                confidence=0.0,
                warnings=[f"Text extraction failed: {e}"],
                requires_review=True,
                review_reasons=["extraction_failed"]
            )

        # =====================================================================
        # Stage 1.5: RAW FIELD EXTRACTION (happens BEFORE classification!)
        # =====================================================================
        # Extract ALL key-value pairs from raw text FIRST
        # This ensures we capture everything regardless of document type
        stage = PipelineStage(name="raw_field_extraction", status="running", started_at=datetime.now())
        stages.append(stage)
        notify_progress(stage)  # Notify: stage started

        raw_fields = {}
        try:
            raw_fields = self._extract_raw_fields(text_result.full_text)
            stage.status = "completed"
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            stage.details = {
                "fields_extracted": len(raw_fields),
                "field_names": list(raw_fields.keys())[:10],  # First 10 for logging
                "method": "regex_patterns",
                "description": "Pre-extraction of key-value pairs using language-agnostic patterns"
            }
            logger.info(f"Raw field extraction complete: {len(raw_fields)} fields")
            notify_progress(stage)  # Notify: stage completed

        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            stage.completed_at = datetime.now()
            logger.warning(f"Raw field extraction failed: {e}")
            notify_progress(stage)  # Notify: stage failed
            raw_fields = {}

        # =====================================================================
        # Stage 2: Find Similar Documents (Vector Store)
        # =====================================================================
        extraction_hints = {}
        similar_docs = []

        if self.retrieval and self.retrieval.vector_store:
            stage = PipelineStage(name="similar_doc_lookup", status="running", started_at=datetime.now())
            stages.append(stage)
            notify_progress(stage)  # Notify: stage started

            try:
                from .adaptive_retrieval import RetrievalStrategy
                # Map "auto" to "simple" (safest default)
                strategy_name = retrieval_strategy or self.default_retrieval_strategy
                if strategy_name == "auto":
                    strategy_name = "simple"
                strategy = RetrievalStrategy(strategy_name)

                retrieval_result = await self.retrieval.retrieve(
                    text=text_result.full_text,
                    strategy=strategy,
                    extraction_prompt="Extract all medical information",
                    layout=text_result.layout
                )

                similar_docs = retrieval_result.similar_docs
                extraction_hints = retrieval_result.extraction_hints

                stage.status = "completed"
                stage.completed_at = datetime.now()
                stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
                stage.details = {
                    "similar_docs_found": len(similar_docs),
                    "strategy_used": retrieval_result.strategy_used,
                    "has_hints": len(extraction_hints.get('field_examples', {})) > 0,
                    "embedding_model": self.embedding_model,
                    "description": "Semantic search for similar documents to guide extraction"
                }

                if similar_docs:
                    logger.info(f"Found {len(similar_docs)} similar documents for extraction hints")
                notify_progress(stage)  # Notify: stage completed

            except Exception as e:
                stage.status = "failed"
                stage.error = str(e)
                stage.completed_at = datetime.now()
                stage.details = {"embedding_model": self.embedding_model}
                logger.warning(f"Similar document lookup failed: {e}")
                notify_progress(stage)  # Notify: stage failed

        # =====================================================================
        # Stage 3: Parallel Content Extraction + Classification
        # =====================================================================
        stage = PipelineStage(name="parallel_extraction", status="running", started_at=datetime.now())
        stages.append(stage)
        notify_progress(stage)  # Notify: stage started

        extraction_result = None
        classification_result = None

        try:
            if self.parallel_extraction and not self.skip_classification:
                # Run extraction and classification in parallel
                # Pass raw_fields to both for better context
                extraction_task = self._extract_content(
                    text_result.full_text,
                    text_result.layout,
                    retrieval_strategy or self.default_retrieval_strategy,
                    extraction_hints,  # Pass hints from similar docs
                    raw_fields  # Pass pre-extracted raw fields
                )
                classification_task = self._classify_document(
                    document_path,
                    text_result.full_text,
                    raw_fields  # Pass raw fields for better classification
                )

                results = await asyncio.gather(
                    extraction_task,
                    classification_task,
                    return_exceptions=True
                )

                # Handle extraction result
                if isinstance(results[0], Exception):
                    logger.error(f"Content extraction failed: {results[0]}")
                    warnings.append(f"Content extraction error: {results[0]}")
                else:
                    extraction_result = results[0]

                # Handle classification result
                if isinstance(results[1], Exception):
                    logger.warning(f"Classification failed: {results[1]}")
                    warnings.append(f"Classification error: {results[1]}")
                else:
                    classification_result = results[1]

            else:
                # Sequential extraction (classification optional)
                extraction_result = await self._extract_content(
                    text_result.full_text,
                    text_result.layout,
                    retrieval_strategy or self.default_retrieval_strategy,
                    extraction_hints,  # Pass hints from similar docs
                    raw_fields  # Pass pre-extracted raw fields
                )

                if not self.skip_classification:
                    try:
                        classification_result = await self._classify_document(
                            document_path,
                            text_result.full_text,
                            raw_fields  # Pass raw fields for better classification
                        )
                    except Exception as e:
                        logger.warning(f"Classification failed: {e}")
                        warnings.append(f"Classification error: {e}")

            stage.status = "completed"
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            stage.details = {
                "extraction_success": extraction_result is not None,
                "classification_success": classification_result is not None,
                "parallel": self.parallel_extraction,
                "llm_model": self.llm_model,
                "description": "LLM-based content extraction and document classification"
            }

            if extraction_result:
                stage.details["test_results_count"] = len(extraction_result.test_results)
                stage.details["medications_count"] = len(extraction_result.medications)
                stage.details["findings_count"] = len(extraction_result.findings)
                stage.details["patient_extracted"] = extraction_result.patient is not None
                stage.details["raw_fields_count"] = len(extraction_result.raw_fields) if extraction_result.raw_fields else 0

            if classification_result:
                stage.details["document_type"] = classification_result.get("type")
                stage.details["classification_confidence"] = classification_result.get("confidence")

            notify_progress(stage)  # Notify: stage completed

        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            stage.completed_at = datetime.now()
            logger.error(f"Parallel extraction failed: {e}")
            warnings.append(f"Extraction pipeline error: {e}")
            notify_progress(stage)  # Notify: stage failed

        # If extraction failed, create empty result
        if extraction_result is None:
            from ..extractors.content_agnostic_extractor import GenericMedicalExtraction
            extraction_result = GenericMedicalExtraction(
                warnings=["Content extraction failed"]
            )
            review_reasons.append("extraction_failed")

        # =====================================================================
        # Stage 3: Merge Classification into Extraction
        # =====================================================================
        # Classification informs but doesn't gate extraction
        if classification_result:
            extraction_result = self._merge_classification(
                extraction_result,
                classification_result
            )

        # =====================================================================
        # Stage 4: Type-Specific Enrichment
        # =====================================================================
        enriched_result = None

        if classification_result:
            doc_type = classification_result.get("type", "unknown")
            classification_confidence = classification_result.get("confidence", 0.0)

            # Only enrich if classification is confident enough
            if classification_confidence >= self.enrichment_confidence_threshold:
                stage = PipelineStage(
                    name=f"{doc_type}_enrichment",
                    status="running",
                    started_at=datetime.now()
                )
                stages.append(stage)
                notify_progress(stage)  # Notify: stage started

                try:
                    enricher = self._get_enricher(doc_type)
                    if enricher:
                        enriched_result = await enricher.enrich(extraction_result)
                        stage.status = "completed"
                        stage.details = {
                            "enricher": doc_type,
                            "description": f"Type-specific enrichment for {doc_type} documents",
                            "llm_model": self.llm_model
                        }
                    else:
                        stage.status = "skipped"
                        stage.details = {"reason": f"No enricher for type: {doc_type}"}

                except Exception as e:
                    stage.status = "failed"
                    stage.error = str(e)
                    logger.warning(f"Enrichment failed for {doc_type}: {e}")
                    warnings.append(f"Enrichment error: {e}")

                stage.completed_at = datetime.now()
                stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
                notify_progress(stage)  # Notify: stage completed/failed

            else:
                logger.info(
                    f"Skipping enrichment: classification confidence "
                    f"({classification_confidence:.2f}) < threshold ({self.enrichment_confidence_threshold})"
                )

        # =====================================================================
        # Stage 4b: Database Validation
        # =====================================================================
        # Surface database validation results as an explicit pipeline step
        # so the user sees it in the Document Processing Pipeline UI.
        if enriched_result and hasattr(enriched_result, 'enrichments'):
            enrichments = enriched_result.enrichments or {}
            validation_summary = enrichments.get('validation_summary', {})

            if validation_summary:
                doc_type = classification_result.get("type", "unknown") if classification_result else "unknown"

                # Build details based on document type
                validation_details = {
                    "description": "Validated extracted data against medical coding databases",
                    "validation_summary": validation_summary,
                }

                total_items = sum(validation_summary.values())
                verified = validation_summary.get('verified', 0) + validation_summary.get('ocr_corrected', 0)
                unverified = validation_summary.get('unverified', 0)

                if doc_type == 'prescription':
                    strength_mismatch = validation_summary.get('strength_mismatch', 0)
                    validation_details.update({
                        "database": "RxNorm",
                        "total_items_checked": total_items,
                        "verified_count": verified,
                        "unverified_count": unverified,
                        "strength_mismatch_count": strength_mismatch,
                        "rxnorm_codes_found": len(enrichments.get('rxnorm_codes', [])),
                        "ocr_corrections": len(enrichments.get('ocr_corrections', [])),
                        "interactions_detected": len(enrichments.get('interactions', [])),
                    })
                elif doc_type == 'lab_report':
                    validation_details.update({
                        "database": "LOINC",
                        "total_items_checked": total_items,
                        "verified_count": verified,
                        "unverified_count": unverified,
                        "loinc_codes_found": len(enrichments.get('loinc_codes', [])),
                    })
                else:
                    # Generic fallback — still show counts
                    db_name = "RxNorm" if enrichments.get('rxnorm_codes') else "LOINC" if enrichments.get('loinc_codes') else "Medical DB"
                    validation_details.update({
                        "database": db_name,
                        "total_items_checked": total_items,
                        "verified_count": verified,
                        "unverified_count": unverified,
                    })

                db_validation_stage = PipelineStage(
                    name="database_validation",
                    status="completed",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    details=validation_details
                )

                # Flag if any items unverified or strength mismatch
                unverified_count = validation_details.get('unverified_count', 0)
                strength_mismatch_count = validation_details.get('strength_mismatch_count', 0)
                warnings = []
                if unverified_count > 0:
                    warnings.append(f"{unverified_count} item(s) could not be verified")
                if strength_mismatch_count > 0:
                    warnings.append(f"{strength_mismatch_count} item(s) have strength mismatch — possible misread")
                if warnings:
                    db_validation_stage.details["warning"] = " | ".join(warnings) + " — requires review"

                stages.append(db_validation_stage)
                notify_progress(db_validation_stage)

                logger.info(
                    f"Database validation stage: {validation_details.get('database', 'N/A')} — "
                    f"{validation_details.get('verified_count', 0)} verified, "
                    f"{validation_details.get('unverified_count', 0)} unverified"
                )

        # =====================================================================
        # Calculate Overall Confidence
        # =====================================================================
        confidence = self._calculate_overall_confidence(
            text_result,
            extraction_result,
            classification_result,
            enriched_result
        )

        # Determine if review is needed
        if confidence < 0.6:
            review_reasons.append("low_confidence")
        if extraction_result and extraction_result.warnings:
            warnings.extend(extraction_result.warnings)
        if len(extraction_result.test_results) == 0 and len(extraction_result.medications) == 0:
            if len(extraction_result.findings) == 0:
                review_reasons.append("no_content_extracted")

        requires_review = len(review_reasons) > 0

        # =====================================================================
        # Stage 5: Store Successful Extraction (Learning)
        # =====================================================================
        # Store high-confidence extractions to vector store for future hints
        if self.store_extractions and confidence >= self.min_store_confidence:
            if self.retrieval and extraction_result:
                try:
                    extracted_values = extraction_result.to_dict()
                    doc_type = classification_result.get("type", "unknown") if classification_result else "unknown"

                    doc_id = await self.retrieval.store_extraction(
                        text=text_result.full_text,
                        extracted_values=extracted_values,
                        template_id=doc_type,
                        source_file=str(document_path),
                        confidence=confidence
                    )

                    if doc_id:
                        logger.info(f"Stored extraction to vector store: {doc_id}")
                        # Add to pipeline stages
                        storage_stage = PipelineStage(
                            name="vector_store",
                            status="completed",
                            started_at=datetime.now(),
                            completed_at=datetime.now(),
                            details={
                                "doc_id": doc_id,
                                "embedding_model": self.embedding_model,
                                "confidence_stored": confidence,
                                "description": "Stored successful extraction for future learning"
                            }
                        )
                        stages.append(storage_stage)
                        notify_progress(storage_stage)

                except Exception as e:
                    logger.warning(f"Failed to store extraction: {e}")

        # Build final result
        total_time = time.time() - start_time

        return ExtractionFirstResult(
            universal_extraction=extraction_result,
            classification=classification_result,
            enriched_extraction=enriched_result,
            text_result=text_result,
            pipeline_stages=stages,
            total_time=total_time,
            confidence=confidence,
            warnings=warnings,
            requires_review=requires_review,
            review_reasons=review_reasons
        )

    async def _process_with_cloud(
        self,
        document_path: Path,
        classification_hint: Optional[str] = None,
        retrieval_strategy: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> ExtractionFirstResult:
        """
        Process document using Azure GPT-4o vision (cloud extraction).

        Stages:
        1. document_intake — File analysis, format detection, page counting
        2. cloud_extraction — Azure GPT-4o vision API call (OCR + extraction + classification)
        3. parallel_extraction — Data structuring + classification in parallel (both parse from same API response)
        4. {type}_enrichment — Type-specific enrichment (if applicable)

        Args:
            document_path: Path to document file
            classification_hint: Optional hint for document type
            retrieval_strategy: Override default retrieval strategy
            progress_callback: Optional callback for progress updates

        Returns:
            ExtractionFirstResult with complete extraction data
        """
        start_time = time.time()
        stages = []
        warnings = []
        review_reasons = []

        document_path = Path(document_path)

        def notify_progress(stage: PipelineStage):
            if progress_callback:
                try:
                    progress_callback(stage, stages)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        # =====================================================================
        # Stage 1: Document Intake — Analyze file format and prepare for API
        # =====================================================================
        stage = PipelineStage(name="document_intake", status="running", started_at=datetime.now())
        stages.append(stage)
        notify_progress(stage)

        file_size_kb = 0
        file_type = document_path.suffix.lower()
        page_count = 1

        try:
            file_size_kb = round(document_path.stat().st_size / 1024, 1)

            # Detect page count for PDFs
            if file_type == '.pdf':
                try:
                    import pymupdf
                    doc = pymupdf.open(document_path)
                    page_count = len(doc)
                    doc.close()
                except Exception:
                    page_count = 1  # fallback

            stage.status = "completed"
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            stage.details = {
                "file_name": document_path.name,
                "file_type": file_type,
                "file_size_kb": file_size_kb,
                "pages": page_count,
                "target_api": "Azure OpenAI GPT-4o",
                "deployment": self.azure_extractor.azure_deployment,
                "description": (
                    f"Analyzed {file_type.upper().strip('.')} document: "
                    f"{page_count} page{'s' if page_count != 1 else ''}, "
                    f"{file_size_kb} KB — preparing for GPT-4o vision API"
                )
            }

            logger.info(f"Document intake: {document_path.name} ({file_type}, {page_count} pages, {file_size_kb} KB)")
            notify_progress(stage)

        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            logger.error(f"Document intake failed: {e}")
            notify_progress(stage)

            return ExtractionFirstResult(
                universal_extraction=None,
                pipeline_stages=stages,
                total_time=time.time() - start_time,
                confidence=0.0,
                warnings=[f"Document intake failed: {e}"],
                requires_review=True,
                review_reasons=["intake_failed"]
            )

        # =====================================================================
        # Stage 2: Cloud Extraction — GPT-4o Vision API Call
        # =====================================================================
        # Single API call: PDF → images → GPT-4o → JSON with classification + extraction
        stage = PipelineStage(name="cloud_extraction", status="running", started_at=datetime.now())
        stages.append(stage)
        notify_progress(stage)

        extraction_result = None

        try:
            extraction_result = await self.azure_extractor.extract_from_file(document_path)

            # Read timing metadata embedded by the azure client
            ext_meta = (extraction_result.raw_fields or {}).get('_extraction_metadata', {})

            stage.status = "completed"
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            stage.details = {
                "method": "azure_gpt4o_vision",
                "deployment": self.azure_extractor.azure_deployment,
                "pages_processed": ext_meta.get('page_count', page_count),
                "conversion_time": ext_meta.get('conversion_time', 0),
                "api_call_time": ext_meta.get('api_call_time', 0),
                "max_tokens": ext_meta.get('max_tokens', self.azure_extractor.max_tokens),
                "temperature": ext_meta.get('temperature', self.azure_extractor.temperature),
                "file_type": file_type,
                "file_name": document_path.name,
                "description": (
                    f"GPT-4o vision processed {ext_meta.get('page_count', page_count)} page image(s) — "
                    f"PDF→image conversion: {ext_meta.get('conversion_time', 0):.1f}s, "
                    f"API call: {ext_meta.get('api_call_time', 0):.1f}s"
                )
            }

            logger.info(
                f"Cloud extraction complete in {stage.duration_seconds:.1f}s "
                f"(conversion: {ext_meta.get('conversion_time', 0):.1f}s, "
                f"API: {ext_meta.get('api_call_time', 0):.1f}s)"
            )

            notify_progress(stage)

            if extraction_result.warnings:
                warnings.extend(extraction_result.warnings)

        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            logger.error(f"Cloud extraction failed: {e}")
            notify_progress(stage)

            return ExtractionFirstResult(
                universal_extraction=None,
                pipeline_stages=stages,
                total_time=time.time() - start_time,
                confidence=0.0,
                warnings=[f"Cloud extraction failed: {e}"],
                requires_review=True,
                review_reasons=["cloud_extraction_failed"]
            )

        # =====================================================================
        # Stage 3: Parallel Extraction + Classification
        # =====================================================================
        # GPT-4o returned both extraction data and classification in one call.
        # We parse them in parallel — data structuring + classification happen
        # simultaneously from the same API response.
        classification_result = None

        stage = PipelineStage(name="parallel_extraction", status="running", started_at=datetime.now())
        stages.append(stage)
        notify_progress(stage)

        extraction_success = False
        classification_success = False
        found_parts = []
        cls_desc = ""

        try:
            # --- Data Structuring (parallel branch 1) ---
            test_count = len(extraction_result.test_results)
            med_count = len(extraction_result.medications)
            finding_count = len(extraction_result.findings)
            proc_count = len(extraction_result.procedures)
            date_count = len(extraction_result.dates)
            provider_count = len(extraction_result.providers)
            org_count = len(extraction_result.organizations)
            raw_count = len(extraction_result.raw_fields) if extraction_result.raw_fields else 0
            has_patient = extraction_result.patient is not None

            if has_patient:
                found_parts.append("patient info")
            if test_count > 0:
                found_parts.append(f"{test_count} test result{'s' if test_count != 1 else ''}")
            if med_count > 0:
                found_parts.append(f"{med_count} medication{'s' if med_count != 1 else ''}")
            if finding_count > 0:
                found_parts.append(f"{finding_count} finding{'s' if finding_count != 1 else ''}")
            if proc_count > 0:
                found_parts.append(f"{proc_count} procedure{'s' if proc_count != 1 else ''}")
            if date_count > 0:
                found_parts.append(f"{date_count} date{'s' if date_count != 1 else ''}")
            if provider_count > 0:
                found_parts.append(f"{provider_count} provider{'s' if provider_count != 1 else ''}")
            if org_count > 0:
                found_parts.append(f"{org_count} organization{'s' if org_count != 1 else ''}")

            extraction_success = True
            logger.info(f"Data structuring: {', '.join(found_parts) if found_parts else 'no data found'}")

            # --- Classification (parallel branch 2) ---
            if not self.skip_classification:
                class_meta = (extraction_result.raw_fields or {}).get('_classification_metadata', {})

                if class_meta and class_meta.get('document_type'):
                    doc_type = class_meta['document_type'].lower().strip().replace(' ', '_')
                    cls_confidence = float(class_meta.get('confidence', 0.8))
                    reasoning = class_meta.get('reasoning', 'GPT-4o vision classification')

                    classification_result = {
                        "type": doc_type,
                        "confidence": cls_confidence,
                        "reasoning": reasoning,
                        "method": "gpt4o_vision",
                    }

                    # Add insurance classification if present
                    ins_class = class_meta.get('insurance_classification', {})
                    if isinstance(ins_class, dict):
                        for key in ('submission_type', 'claim_type', 'line_of_benefits', 'benefit_type'):
                            val = ins_class.get(key)
                            if val and str(val).lower() not in ('null', 'none', ''):
                                classification_result[key] = val

                    cls_desc = f"Identified as {doc_type} (confidence: {cls_confidence:.0%})"
                    if classification_result.get('claim_type'):
                        cls_desc += f" — {classification_result['submission_type'] or 'claim'}: {classification_result['claim_type']}"
                    if classification_result.get('line_of_benefits'):
                        cls_desc += f" → {classification_result['line_of_benefits']}"
                    if classification_result.get('benefit_type'):
                        cls_desc += f" → {classification_result['benefit_type']}"

                    classification_success = True
                    logger.info(f"Classification: {cls_desc}")
                else:
                    classification_result = {
                        "type": "unknown",
                        "confidence": 0.3,
                        "reasoning": "GPT-4o did not return classification",
                        "method": "gpt4o_vision_missing"
                    }
                    cls_desc = "GPT-4o response did not include document classification"
                    classification_success = True
                    warnings.append("GPT-4o did not return document classification")

            ext_meta = (extraction_result.raw_fields or {}).get('_extraction_metadata', {})

            stage.status = "completed"
            stage.completed_at = datetime.now()
            stage.duration_seconds = ext_meta.get('structuring_time', (stage.completed_at - stage.started_at).total_seconds())
            stage.details = {
                "parallel": True,
                "extraction_success": extraction_success,
                "classification_success": classification_success,
                "deployment": self.azure_extractor.azure_deployment,
                # Data structuring details
                "patient_extracted": has_patient,
                "test_results_count": test_count,
                "medications_count": med_count,
                "findings_count": finding_count,
                "procedures_count": proc_count,
                "dates_count": date_count,
                "providers_count": provider_count,
                "organizations_count": org_count,
                "raw_fields_count": raw_count,
                "extraction_confidence": extraction_result.extraction_confidence,
                # Classification details
                "document_type": classification_result.get("type", "unknown") if classification_result else "unknown",
                "classification_confidence": classification_result.get("confidence", 0.0) if classification_result else 0.0,
                "reasoning": classification_result.get("reasoning", "") if classification_result else "",
                "submission_type": classification_result.get("submission_type") if classification_result else None,
                "claim_type": classification_result.get("claim_type") if classification_result else None,
                "line_of_benefits": classification_result.get("line_of_benefits") if classification_result else None,
                "benefit_type": classification_result.get("benefit_type") if classification_result else None,
                "description": (
                    f"Structured {len(found_parts)} data categories"
                    + (f" ({', '.join(found_parts)})" if found_parts else "")
                    + (f" | {cls_desc}" if cls_desc else "")
                )
            }

            notify_progress(stage)

        except Exception as e:
            stage.status = "failed"
            stage.error = str(e)
            stage.completed_at = datetime.now()
            stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
            logger.warning(f"Parallel extraction+classification failed: {e}")
            warnings.append(f"Processing error: {e}")
            notify_progress(stage)

        # If extraction failed, create empty result
        if extraction_result is None:
            from ..extractors.content_agnostic_extractor import GenericMedicalExtraction
            extraction_result = GenericMedicalExtraction(
                warnings=["Cloud extraction failed"]
            )
            review_reasons.append("extraction_failed")

        # Merge classification into extraction
        if classification_result:
            extraction_result = self._merge_classification(extraction_result, classification_result)

        # =====================================================================
        # Stage 4: Type-Specific Enrichment (same as local path)
        # =====================================================================
        enriched_result = None

        if classification_result:
            doc_type = classification_result.get("type", "unknown")
            classification_confidence = classification_result.get("confidence", 0.0)

            if classification_confidence >= self.enrichment_confidence_threshold:
                stage = PipelineStage(
                    name=f"{doc_type}_enrichment",
                    status="running",
                    started_at=datetime.now()
                )
                stages.append(stage)
                notify_progress(stage)

                try:
                    enricher = self._get_enricher(doc_type)
                    if enricher:
                        enriched_result = await enricher.enrich(extraction_result)
                        stage.status = "completed"
                        stage.details = {
                            "enricher": doc_type,
                            "description": f"Type-specific enrichment for {doc_type} documents"
                        }
                    else:
                        stage.status = "skipped"
                        stage.details = {
                            "reason": f"No enricher for type: {doc_type}",
                            "description": f"No specialized enricher available for '{doc_type}' documents — skipping"
                        }

                except Exception as e:
                    stage.status = "failed"
                    stage.error = str(e)
                    logger.warning(f"Enrichment failed for {doc_type}: {e}")
                    warnings.append(f"Enrichment error: {e}")

                stage.completed_at = datetime.now()
                stage.duration_seconds = (stage.completed_at - stage.started_at).total_seconds()
                notify_progress(stage)

        # =====================================================================
        # Stage 4b: Database Validation (cloud path)
        # =====================================================================
        if enriched_result and hasattr(enriched_result, 'enrichments'):
            enrichments = enriched_result.enrichments or {}
            validation_summary = enrichments.get('validation_summary', {})

            if validation_summary:
                doc_type = classification_result.get("type", "unknown") if classification_result else "unknown"

                validation_details = {
                    "description": "Validated extracted data against medical coding databases",
                    "validation_summary": validation_summary,
                }

                total_items = sum(validation_summary.values())
                verified = validation_summary.get('verified', 0) + validation_summary.get('ocr_corrected', 0)
                unverified = validation_summary.get('unverified', 0)

                if doc_type == 'prescription':
                    strength_mismatch = validation_summary.get('strength_mismatch', 0)
                    validation_details.update({
                        "database": "RxNorm",
                        "total_items_checked": total_items,
                        "verified_count": verified,
                        "unverified_count": unverified,
                        "strength_mismatch_count": strength_mismatch,
                        "rxnorm_codes_found": len(enrichments.get('rxnorm_codes', [])),
                        "ocr_corrections": len(enrichments.get('ocr_corrections', [])),
                        "interactions_detected": len(enrichments.get('interactions', [])),
                    })
                elif doc_type == 'lab_report':
                    validation_details.update({
                        "database": "LOINC",
                        "total_items_checked": total_items,
                        "verified_count": verified,
                        "unverified_count": unverified,
                        "loinc_codes_found": len(enrichments.get('loinc_codes', [])),
                    })
                else:
                    db_name = "RxNorm" if enrichments.get('rxnorm_codes') else "LOINC" if enrichments.get('loinc_codes') else "Medical DB"
                    validation_details.update({
                        "database": db_name,
                        "total_items_checked": total_items,
                        "verified_count": verified,
                        "unverified_count": unverified,
                    })

                db_validation_stage = PipelineStage(
                    name="database_validation",
                    status="completed",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    details=validation_details
                )

                # Flag if any items unverified or strength mismatch
                unverified_count = validation_details.get('unverified_count', 0)
                strength_mismatch_count = validation_details.get('strength_mismatch_count', 0)
                warnings = []
                if unverified_count > 0:
                    warnings.append(f"{unverified_count} item(s) could not be verified")
                if strength_mismatch_count > 0:
                    warnings.append(f"{strength_mismatch_count} item(s) have strength mismatch — possible misread")
                if warnings:
                    db_validation_stage.details["warning"] = " | ".join(warnings) + " — requires review"

                stages.append(db_validation_stage)
                notify_progress(db_validation_stage)

                logger.info(
                    f"Database validation stage (cloud): {validation_details.get('database', 'N/A')} — "
                    f"{validation_details.get('verified_count', 0)} verified, "
                    f"{validation_details.get('unverified_count', 0)} unverified"
                )

        # =====================================================================
        # Calculate Overall Confidence
        # =====================================================================
        confidence = extraction_result.extraction_confidence if extraction_result else 0.0

        if classification_result and classification_result.get("confidence", 0) > 0:
            confidence = (confidence + classification_result["confidence"]) / 2

        if confidence < 0.6:
            review_reasons.append("low_confidence")
        if extraction_result and extraction_result.warnings:
            warnings.extend(extraction_result.warnings)
        # Check if any meaningful content was extracted (including procedures for dental/insurance)
        has_content = (
            len(extraction_result.test_results) > 0
            or len(extraction_result.medications) > 0
            or len(extraction_result.findings) > 0
            or len(extraction_result.procedures) > 0
            or len(extraction_result.raw_fields) > 3  # insurance docs may only have raw fields
        )
        if not has_content:
            review_reasons.append("no_content_extracted")

        requires_review = len(review_reasons) > 0

        # Build final result
        total_time = time.time() - start_time

        return ExtractionFirstResult(
            universal_extraction=extraction_result,
            classification=classification_result,
            enriched_extraction=enriched_result,
            text_result=None,  # No local text extraction for cloud path
            pipeline_stages=stages,
            total_time=total_time,
            confidence=confidence,
            warnings=warnings,
            requires_review=requires_review,
            review_reasons=review_reasons
        )

    async def _extract_text(self, document_path: Path, page_progress_callback=None):
        """Extract text using UniversalTextExtractor."""
        if self._text_extractor is None:
            from ..extractors.universal_text_extractor import UniversalTextExtractor
            self._text_extractor = UniversalTextExtractor(self.config)

        return await self._text_extractor.extract(document_path, progress_callback=page_progress_callback)

    async def _extract_content(
        self,
        text: str,
        layout: Any,
        retrieval_strategy: str,
        extraction_hints: Dict[str, Any] = None,
        raw_fields: Dict[str, Any] = None
    ):
        """
        Extract medical content using ContentAgnosticExtractor.

        Args:
            text: Document text
            layout: Layout information
            retrieval_strategy: Retrieval strategy to use
            extraction_hints: Hints from similar documents (vector store)
            raw_fields: Pre-extracted key-value pairs from raw text
        """
        if self._content_extractor is None:
            from ..extractors.content_agnostic_extractor import ContentAgnosticExtractor
            logger.info(f"Creating ContentAgnosticExtractor with max_text_length={self.config.get('max_text_length', 'NOT SET')}")
            self._content_extractor = ContentAgnosticExtractor(
                config=self.config,
                prompt_manager=self.prompt_manager,
                retrieval=self.retrieval
            )

        result = await self._content_extractor.extract(
            text=text,
            layout=layout,
            retrieval_strategy=retrieval_strategy,
            extraction_hints=extraction_hints
        )

        # Merge pre-extracted raw_fields into the result
        # These were extracted BEFORE classification, so they're guaranteed to be there
        if raw_fields and hasattr(result, 'raw_fields'):
            # Merge: pre-extracted fields take precedence (extracted first)
            merged_fields = {**result.raw_fields, **raw_fields}
            result.raw_fields = merged_fields
            logger.info(f"Merged {len(raw_fields)} pre-extracted fields into result")

        return result

    async def _classify_document(
        self,
        document_path: Path,
        text: str,
        raw_fields: Dict[str, Any] = None
    ):
        """
        Classify document using DocumentClassifier.

        Args:
            document_path: Path to document
            text: Extracted text
            raw_fields: Pre-extracted key-value pairs (for better classification)
        """
        if self._classifier is None:
            from ..classifiers.document_classifier import DocumentClassifier
            self._classifier = DocumentClassifier(self.config)

        # Create a minimal context for the classifier
        from . import ProcessingContext
        context = ProcessingContext(document_path=document_path)
        context.raw_text = text

        # Pass raw_fields to context so classifier can use them
        if raw_fields:
            context.sections['_raw_fields'] = raw_fields
            # Also add key fields directly for easier access
            if 'claim_type' in raw_fields:
                context.sections['claim_type'] = raw_fields['claim_type']
            if 'benefit_type' in raw_fields:
                context.sections['benefit_type'] = raw_fields['benefit_type']
            if 'service_type' in raw_fields:
                context.sections['service_type'] = raw_fields['service_type']

        return await self._classifier.execute(context)

    def _extract_raw_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract ALL key-value pairs from raw text.

        This runs BEFORE classification and LLM extraction to ensure
        no data is lost regardless of document type.

        Returns:
            Dictionary of field_name -> value pairs
        """
        import re

        raw_fields = {}

        if not text:
            return raw_fields

        # Split text into lines for cleaner processing
        lines = text.split('\n')

        # Pattern 1: Process line by line for "Label : Value" patterns
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Match "Label : Value" or "Label: Value"
            match = re.match(r'^([^:]+?)\s*:\s*(.+)$', line)
            if match:
                label = match.group(1).strip()
                value = match.group(2).strip()

                # Clean bilingual labels - take the English part
                if ' / ' in label:
                    label = label.split(' / ')[0].strip()

                # Clean bilingual values
                if ' / ' in value and len(value.split(' / ')) == 2:
                    parts = value.split(' / ')
                    if not any(c in parts[0] for c in ['é', 'è', 'à', 'ç']):
                        value = parts[0].strip()

                # Skip if label is too long or value too short
                if len(label) <= 50 and len(value) >= 2 and value.lower() not in ['n/a', 'na', '']:
                    key = label.lower().replace(' ', '_').replace('-', '_')
                    key = re.sub(r'[^a-z0-9_]', '', key)
                    key = re.sub(r'_+', '_', key).strip('_')

                    if key and len(key) >= 2 and key not in raw_fields:
                        raw_fields[key] = value

        # Pattern 2: Invoice/Reference numbers
        id_patterns = [
            (r'Invoice\s*#?\s*[:.]?\s*([A-Z0-9\-]+)', 'invoice_number'),
            (r'Reference\s*#?\s*[:.]?\s*([A-Z0-9\-]+)', 'reference_number'),
            (r'Receipt\s*#?\s*[:.]?\s*([A-Z0-9\-]+)', 'receipt_number'),
            (r'Account\s*#?\s*[:.]?\s*([A-Z0-9\-]+)', 'account_number'),
            (r'Member\s*(?:ID|#)?\s*[:.]?\s*([A-Z0-9\-]+)', 'member_id'),
            (r'Policy\s*#?\s*[:.]?\s*([A-Z0-9\-]+)', 'policy_number'),
        ]

        for pattern, field_name in id_patterns:
            if field_name not in raw_fields:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    raw_fields[field_name] = match.group(1)

        # Pattern 3: Dollar amounts
        amount_matches = re.findall(r'\$\s*([\d,]+\.?\d*)', text)
        if amount_matches:
            amounts = list(set(amount_matches))
            for i, amt in enumerate(amounts[:5]):
                key = f'amount_{i+1}' if i > 0 else 'total_amount'
                if key not in raw_fields:
                    raw_fields[key] = f"${amt}"

        # Pattern 4: Dates
        date_patterns = [
            (r'(\d{4}-\d{2}-\d{2})', 'date_iso'),
            (r'(\d{1,2}/\d{1,2}/\d{4})', 'date_slash'),
            (r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})', 'date_full'),
        ]

        for pattern, field_name in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for i, dt in enumerate(matches[:3]):
                    key = f'{field_name}_{i+1}' if i > 0 else field_name
                    if key not in raw_fields:
                        raw_fields[key] = dt

        # Pattern 5: Email addresses
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if emails:
            raw_fields['email'] = emails[0]
            if len(emails) > 1:
                raw_fields['emails'] = emails

        # Pattern 6: Phone numbers
        phones = re.findall(r'\b\d{3}[\s\-.]?\d{3}[\s\-.]?\d{4}\b', text)
        if phones:
            raw_fields['phone'] = phones[0]

        # Pattern 7: Specific medical/insurance fields
        specific_patterns = [
            (r'Type of claim[^:]*:\s*([A-Za-z\s]+?)(?:\s*/|\s*$|\n)', 'claim_type'),
            (r'(?:Benefit|Prestations?)[^:]*:\s*([A-Za-z\s]+?)(?:\s*/|\s*$|\n)', 'benefit_type'),
            (r'Participant[^:]*:\s*(\S+\s+[A-Z][A-Z\s]+)', 'participant'),
            (r'Submission Date[^:]*:\s*(\d{4}-\d{2}-\d{2})', 'submission_date'),
        ]

        for pattern, field_name in specific_patterns:
            if field_name not in raw_fields:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value and value.lower() not in ['n/a', 'na', 'none', '']:
                        raw_fields[field_name] = value

        return raw_fields

    def _merge_classification(
        self,
        extraction: Any,
        classification: Dict[str, Any]
    ) -> Any:
        """
        Merge classification info into extraction result.

        Classification can inform categorization of extracted items.
        """
        doc_type = classification.get("type", "unknown")

        # Update test result categories based on classification
        if doc_type == "lab":
            for result in extraction.test_results:
                if not result.category:
                    result.category = "lab"

        elif doc_type == "prescription":
            # Medications are already extracted, but we can boost confidence
            for med in extraction.medications:
                if med.confidence < 0.9:
                    med.confidence = min(med.confidence + 0.1, 0.95)

        elif doc_type == "radiology":
            for finding in extraction.findings:
                if not finding.category:
                    finding.category = "impression"

        return extraction

    def _get_enricher(self, doc_type: str):
        """Get type-specific enricher."""
        if self._enrichers is None:
            self._enrichers = {}

            # Lazy load enrichers
            try:
                from ..enrichers.lab_enricher import LabEnricher
                self._enrichers["lab"] = LabEnricher(self.config)
            except ImportError:
                pass

            try:
                from ..enrichers.prescription_enricher import PrescriptionEnricher
                self._enrichers["prescription"] = PrescriptionEnricher(self.config)
            except ImportError:
                pass

            try:
                from ..enrichers.radiology_enricher import RadiologyEnricher
                self._enrichers["radiology"] = RadiologyEnricher(self.config)
            except ImportError:
                pass

            try:
                from ..enrichers.pathology_enricher import PathologyEnricher
                self._enrichers["pathology"] = PathologyEnricher(self.config)
            except ImportError:
                pass

        return self._enrichers.get(doc_type)

    def _calculate_overall_confidence(
        self,
        text_result: Any,
        extraction_result: Any,
        classification_result: Optional[Dict],
        enriched_result: Any
    ) -> float:
        """Calculate overall pipeline confidence.

        Weighted by importance:
        - Extraction confidence (what we actually extracted) is the primary signal
        - Classification confidence is secondary
        - Text extraction and enrichment are minor adjustments
        """
        # (score, weight) pairs
        weighted = []

        # Extraction confidence — most important, this is the actual result quality
        if extraction_result and extraction_result.extraction_confidence > 0:
            weighted.append((extraction_result.extraction_confidence, 5))

        # Classification confidence
        if classification_result and classification_result.get("confidence", 0) > 0:
            weighted.append((classification_result["confidence"], 2))

        # Text extraction confidence (OCR/VLM quality)
        if text_result and text_result.confidence > 0:
            weighted.append((text_result.confidence, 1))

        # Enrichment confidence
        if enriched_result and hasattr(enriched_result, 'enrichment_confidence'):
            if enriched_result.enrichment_confidence > 0:
                weighted.append((enriched_result.enrichment_confidence, 2))

        if not weighted:
            return 0.5

        total_weight = sum(w for _, w in weighted)
        return round(sum(s * w for s, w in weighted) / total_weight, 3)


# Convenience function
async def process_document_extraction_first(
    document_path: Path,
    config: Dict[str, Any] = None,
    retrieval_strategy: str = "router"
) -> ExtractionFirstResult:
    """
    Convenience function for extraction-first processing.

    Args:
        document_path: Path to document file
        config: Optional configuration
        retrieval_strategy: Retrieval strategy to use

    Returns:
        ExtractionFirstResult with complete extraction data
    """
    pipeline = ExtractionFirstPipeline(config or {})
    return await pipeline.process(
        document_path=document_path,
        retrieval_strategy=retrieval_strategy
    )
