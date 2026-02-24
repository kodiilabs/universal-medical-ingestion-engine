# ============================================================================
# src/medical_ingestion/core/adaptive_retrieval.py
# ============================================================================
"""
Adaptive Retrieval System

Inspired by Unstract's 7 retrieval strategies. Implements multiple approaches
for selecting and presenting context to the LLM for extraction.

Key features (like Unstract):
- Vector store integration for embedding-based similarity
- Multiple retrieval strategies with automatic selection
- Similar document retrieval to guide extraction
- Configurable chunking with semantic boundaries

Strategies:
1. SIMPLE - Full text as context (best for short documents)
2. ROUTER - LLM selects best strategy based on document characteristics
3. FUSION - Run multiple strategies and merge results
4. CHUNKED - Split into chunks, use embeddings to select relevant ones
5. VECTOR - Use vector store embeddings for semantic chunk selection

Future strategies (can be added):
6. SUBQUESTION - Break complex queries into sub-questions
7. RECURSIVE - Iteratively refine extraction
8. AUTOMERGING - Boundary-aware chunk merging
"""

import asyncio
import logging
import re
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    SIMPLE = "simple"       # Full text as context
    ROUTER = "router"       # LLM selects best strategy
    FUSION = "fusion"       # Combine multiple strategies
    CHUNKED = "chunked"     # Split into chunks, keyword-based selection
    VECTOR = "vector"       # Use embeddings for semantic chunk selection


@dataclass
class RetrievalContext:
    """Context for retrieval operations."""
    text: str
    layout: Optional[Any] = None  # LayoutInfo from UniversalTextExtractor
    similar_docs: List[Dict] = field(default_factory=list)  # From vector store
    strategy: RetrievalStrategy = RetrievalStrategy.SIMPLE
    chunk_size: int = 2000
    chunk_overlap: int = 200
    # Embedding-related fields
    chunk_embeddings: List[Tuple[str, Any]] = field(default_factory=list)  # [(chunk_text, embedding)]
    use_vector_store: bool = True  # Whether to use vector store for similarity


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    context_text: str
    strategy_used: str
    chunks_selected: int = 0
    total_chunks: int = 0
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Vector store related
    similar_docs: List[Dict] = field(default_factory=list)  # Similar documents found
    extraction_hints: Dict[str, Any] = field(default_factory=dict)  # From similar docs


class AdaptiveRetrieval:
    """
    Adaptive retrieval for medical document extraction.

    Integrates with VectorStore for embedding-based similarity:
    - Find similar documents to guide extraction
    - Use embeddings for semantic chunk selection
    - Store successful extractions for future reference

    Usage:
        retrieval = AdaptiveRetrieval()
        result = await retrieval.retrieve(
            text="...",
            strategy=RetrievalStrategy.ROUTER,
            extraction_prompt="Extract all lab values"
        )
        # Use result.context_text for LLM extraction
        # result.similar_docs contains similar documents
        # result.extraction_hints contains hints from similar docs
    """

    def __init__(self, config: Dict[str, Any] = None, vector_store: 'VectorStore' = None):
        self.config = config or {}
        self._llm_client = None
        self._vector_store = vector_store

        # Configuration
        self.default_strategy = RetrievalStrategy(
            self.config.get('default_strategy', 'simple')
        )
        self.max_context_length = self.config.get('max_context_length', 8000)
        self.chunk_size = self.config.get('chunk_size', 2000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)

        # Vector store settings
        self.use_vector_store = self.config.get('use_vector_store', True)
        self.min_similarity = self.config.get('min_similarity', 0.6)
        self.top_k_similar = self.config.get('top_k_similar', 3)
        self.top_k_chunks = self.config.get('top_k_chunks', 5)

    @property
    def vector_store(self) -> Optional['VectorStore']:
        """Lazy load vector store if not provided."""
        if self._vector_store is None and self.use_vector_store:
            try:
                from .vector_store import get_vector_store
                self._vector_store = get_vector_store(self.config)
            except Exception as e:
                logger.warning(f"Could not initialize vector store: {e}")
                self._vector_store = None
        return self._vector_store

    async def retrieve(
        self,
        text: str,
        strategy: RetrievalStrategy = None,
        extraction_prompt: str = None,
        layout: Optional[Any] = None
    ) -> RetrievalResult:
        """
        Execute retrieval strategy and return context.

        Args:
            text: Full document text
            strategy: Retrieval strategy to use (defaults to config)
            extraction_prompt: The extraction prompt (used by router)
            layout: Optional layout info

        Returns:
            RetrievalResult with context text ready for LLM
        """
        strategy = strategy or self.default_strategy

        ctx = RetrievalContext(
            text=text,
            layout=layout,
            strategy=strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            use_vector_store=self.use_vector_store
        )

        # Find similar documents from vector store
        similar_docs = []
        extraction_hints = {}
        if self.vector_store and self.use_vector_store:
            try:
                similar_docs = await self._find_similar_documents(text)
                extraction_hints = self._extract_hints_from_similar(similar_docs)
                ctx.similar_docs = similar_docs
                logger.info(f"Found {len(similar_docs)} similar documents")
            except Exception as e:
                logger.warning(f"Similar document lookup failed: {e}")

        logger.info(f"Executing retrieval strategy: {strategy.value}")

        if strategy == RetrievalStrategy.SIMPLE:
            result = await self._simple_retrieval(ctx)
        elif strategy == RetrievalStrategy.ROUTER:
            result = await self._router_retrieval(ctx, extraction_prompt)
        elif strategy == RetrievalStrategy.FUSION:
            result = await self._fusion_retrieval(ctx, extraction_prompt)
        elif strategy == RetrievalStrategy.CHUNKED:
            result = await self._chunked_retrieval(ctx, extraction_prompt)
        elif strategy == RetrievalStrategy.VECTOR:
            result = await self._vector_retrieval(ctx, extraction_prompt)
        else:
            # Fallback to simple
            result = await self._simple_retrieval(ctx)

        # Attach similar docs and hints to result
        result.similar_docs = similar_docs
        result.extraction_hints = extraction_hints

        return result

    async def _find_similar_documents(self, text: str) -> List[Dict[str, Any]]:
        """
        Find similar documents from vector store.

        Returns documents with their previous extractions to guide current extraction.
        """
        if not self.vector_store:
            return []

        # Use first 2000 chars for similarity search
        search_text = text[:2000] if len(text) > 2000 else text

        similar = await self.vector_store.find_similar(
            text=search_text,
            top_k=self.top_k_similar,
            min_similarity=self.min_similarity
        )

        return similar

    def _extract_hints_from_similar(self, similar_docs: List[Dict]) -> Dict[str, Any]:
        """
        Extract hints from similar documents to guide extraction.

        This is key to Unstract's approach: use successful extractions
        from similar documents to inform current extraction.
        """
        hints = {
            'field_examples': {},
            'expected_fields': set(),
            'common_patterns': [],
            'confidence_boost': 0.0
        }

        if not similar_docs:
            return hints

        for doc in similar_docs:
            similarity = doc.get('similarity', 0)
            extracted_values = doc.get('extracted_values', {})

            # Collect field examples from similar docs
            for field_name, value in extracted_values.items():
                if field_name not in hints['field_examples']:
                    hints['field_examples'][field_name] = []
                hints['field_examples'][field_name].append({
                    'value': value,
                    'similarity': similarity
                })
                hints['expected_fields'].add(field_name)

        # Convert set to list for JSON serialization
        hints['expected_fields'] = list(hints['expected_fields'])

        # Calculate confidence boost based on similarity
        if similar_docs:
            avg_similarity = sum(d.get('similarity', 0) for d in similar_docs) / len(similar_docs)
            hints['confidence_boost'] = min(avg_similarity * 0.1, 0.1)

        return hints

    async def _simple_retrieval(self, ctx: RetrievalContext) -> RetrievalResult:
        """
        Simple retrieval: Use full text as context.

        Best for: Short documents (<8000 chars)
        """
        text = ctx.text

        # Truncate if too long
        if len(text) > self.max_context_length:
            text = text[:self.max_context_length] + "\n...[truncated]..."

        return RetrievalResult(
            context_text=text,
            strategy_used="simple",
            chunks_selected=1,
            total_chunks=1,
            relevance_score=1.0,
            metadata={"original_length": len(ctx.text), "truncated": len(ctx.text) > self.max_context_length}
        )

    async def _router_retrieval(
        self,
        ctx: RetrievalContext,
        extraction_prompt: str = None
    ) -> RetrievalResult:
        """
        Router retrieval: LLM selects best strategy.

        Analyzes document characteristics and chooses optimal approach.
        """
        # For short documents, just use simple
        if len(ctx.text) <= self.max_context_length:
            return await self._simple_retrieval(ctx)

        # Analyze document to choose strategy
        doc_analysis = self._analyze_document(ctx)

        # Decision logic based on analysis
        # Prefer VECTOR strategy when vector store is available and we have similar docs
        has_similar_docs = len(ctx.similar_docs) > 0

        if doc_analysis['is_short']:
            selected_strategy = RetrievalStrategy.SIMPLE
        elif has_similar_docs and self.vector_store:
            # Use vector retrieval when we have similar documents for guidance
            selected_strategy = RetrievalStrategy.VECTOR
        elif doc_analysis['has_clear_sections']:
            # Use chunked retrieval with section awareness
            selected_strategy = RetrievalStrategy.CHUNKED
        elif doc_analysis['is_complex']:
            # Use fusion for complex documents
            selected_strategy = RetrievalStrategy.FUSION
        else:
            # Default: use vector if available, else chunked
            if self.vector_store:
                selected_strategy = RetrievalStrategy.VECTOR
            else:
                selected_strategy = RetrievalStrategy.CHUNKED

        logger.info(
            f"Router selected strategy: {selected_strategy.value} "
            f"(doc_length={len(ctx.text)}, sections={doc_analysis['section_count']}, "
            f"similar_docs={len(ctx.similar_docs)})"
        )

        # Execute selected strategy
        if selected_strategy == RetrievalStrategy.SIMPLE:
            result = await self._simple_retrieval(ctx)
        elif selected_strategy == RetrievalStrategy.CHUNKED:
            result = await self._chunked_retrieval(ctx, extraction_prompt)
        elif selected_strategy == RetrievalStrategy.FUSION:
            result = await self._fusion_retrieval(ctx, extraction_prompt)
        elif selected_strategy == RetrievalStrategy.VECTOR:
            result = await self._vector_retrieval(ctx, extraction_prompt)
        else:
            result = await self._simple_retrieval(ctx)

        # Update strategy used
        result.strategy_used = f"router->{selected_strategy.value}"
        result.metadata['router_analysis'] = doc_analysis

        return result

    async def _fusion_retrieval(
        self,
        ctx: RetrievalContext,
        extraction_prompt: str = None
    ) -> RetrievalResult:
        """
        Fusion retrieval: Combine multiple strategies.

        Runs simple + chunked and merges results for comprehensive coverage.
        """
        # Run multiple strategies in parallel
        results = await asyncio.gather(
            self._simple_retrieval(ctx),
            self._chunked_retrieval(ctx, extraction_prompt),
            return_exceptions=True
        )

        # Collect valid results
        valid_results = []
        for r in results:
            if not isinstance(r, Exception):
                valid_results.append(r)

        if not valid_results:
            # Fallback to simple if everything failed
            return await self._simple_retrieval(ctx)

        # Merge contexts
        # For fusion, we use the chunked result but prepend document summary
        if len(valid_results) >= 2:
            simple_result = valid_results[0]
            chunked_result = valid_results[1]

            # Create summary from start and end of document
            summary_length = 500
            doc_summary = f"[Document Start]\n{ctx.text[:summary_length]}\n...\n"
            if len(ctx.text) > summary_length * 2:
                doc_summary += f"...\n{ctx.text[-summary_length:]}\n[Document End]\n\n"

            # Combine: summary + relevant chunks
            combined_context = f"{doc_summary}[Relevant Sections]\n{chunked_result.context_text}"

            # Truncate if needed
            if len(combined_context) > self.max_context_length:
                combined_context = combined_context[:self.max_context_length]

            return RetrievalResult(
                context_text=combined_context,
                strategy_used="fusion",
                chunks_selected=chunked_result.chunks_selected + 1,
                total_chunks=chunked_result.total_chunks,
                relevance_score=max(r.relevance_score for r in valid_results),
                metadata={
                    "strategies_combined": [r.strategy_used for r in valid_results],
                    "summary_included": True
                }
            )
        else:
            # Only one strategy succeeded
            result = valid_results[0]
            result.strategy_used = f"fusion->{result.strategy_used}"
            return result

    async def _chunked_retrieval(
        self,
        ctx: RetrievalContext,
        extraction_prompt: str = None
    ) -> RetrievalResult:
        """
        Chunked retrieval: Split into chunks and select relevant ones.

        Uses keyword matching to find most relevant chunks for the extraction task.
        """
        # Split text into chunks
        chunks = self._split_into_chunks(
            ctx.text,
            chunk_size=ctx.chunk_size,
            overlap=ctx.chunk_overlap
        )

        if not chunks:
            return await self._simple_retrieval(ctx)

        # Score chunks for relevance
        scored_chunks = self._score_chunks(chunks, extraction_prompt)

        # Select top chunks that fit in context
        selected_chunks = self._select_chunks(
            scored_chunks,
            max_length=self.max_context_length
        )

        # Combine selected chunks
        context_text = "\n\n---\n\n".join([c['text'] for c in selected_chunks])

        # Calculate average relevance
        avg_relevance = (
            sum(c['score'] for c in selected_chunks) / len(selected_chunks)
            if selected_chunks else 0.0
        )

        return RetrievalResult(
            context_text=context_text,
            strategy_used="chunked",
            chunks_selected=len(selected_chunks),
            total_chunks=len(chunks),
            relevance_score=avg_relevance,
            metadata={
                "chunk_scores": [(c['index'], c['score']) for c in selected_chunks[:5]]
            }
        )

    async def _vector_retrieval(
        self,
        ctx: RetrievalContext,
        extraction_prompt: str = None
    ) -> RetrievalResult:
        """
        Vector retrieval: Use embeddings for semantic chunk selection.

        This is the Unstract-inspired approach:
        1. Split document into chunks
        2. Compute embeddings for each chunk
        3. Compute embedding for extraction prompt/query
        4. Select chunks with highest similarity to query
        5. Include hints from similar documents

        Falls back to keyword-based chunked retrieval if vector store unavailable.
        """
        if not self.vector_store:
            logger.info("Vector store unavailable, falling back to chunked retrieval")
            return await self._chunked_retrieval(ctx, extraction_prompt)

        # Split text into chunks
        chunks = self._split_into_chunks(
            ctx.text,
            chunk_size=ctx.chunk_size,
            overlap=ctx.chunk_overlap
        )

        if not chunks:
            return await self._simple_retrieval(ctx)

        # Compute embeddings for chunks and query
        try:
            scored_chunks = await self._score_chunks_by_embedding(
                chunks,
                extraction_prompt or "Extract all medical information"
            )
        except Exception as e:
            logger.warning(f"Embedding-based scoring failed: {e}, falling back to keyword")
            scored_chunks = self._score_chunks(chunks, extraction_prompt)

        # Select top chunks that fit in context
        selected_chunks = self._select_chunks(
            scored_chunks,
            max_length=self.max_context_length
        )

        # Build context with hints from similar documents
        context_parts = []

        # Add extraction hints from similar documents
        if ctx.similar_docs:
            hints_text = self._format_extraction_hints(ctx.similar_docs)
            if hints_text:
                context_parts.append(f"[Extraction Hints from Similar Documents]\n{hints_text}\n")

        # Add selected chunks
        chunks_text = "\n\n---\n\n".join([c['text'] for c in selected_chunks])
        context_parts.append(f"[Document Content]\n{chunks_text}")

        context_text = "\n".join(context_parts)

        # Truncate if needed
        if len(context_text) > self.max_context_length:
            context_text = context_text[:self.max_context_length]

        # Calculate average relevance
        avg_relevance = (
            sum(c['score'] for c in selected_chunks) / len(selected_chunks)
            if selected_chunks else 0.0
        )

        return RetrievalResult(
            context_text=context_text,
            strategy_used="vector",
            chunks_selected=len(selected_chunks),
            total_chunks=len(chunks),
            relevance_score=avg_relevance,
            metadata={
                "chunk_scores": [(c['index'], c['score']) for c in selected_chunks[:5]],
                "embedding_based": True,
                "similar_docs_count": len(ctx.similar_docs)
            }
        )

    async def _score_chunks_by_embedding(
        self,
        chunks: List[str],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Score chunks using embedding similarity.

        Computes cosine similarity between query embedding and chunk embeddings.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not available")

        # Compute query embedding
        query_embedding = self.vector_store._compute_embedding(query)

        scored = []
        for i, chunk in enumerate(chunks):
            # Compute chunk embedding
            chunk_embedding = self.vector_store._compute_embedding(chunk)

            # Calculate cosine similarity
            similarity = self.vector_store._cosine_similarity(query_embedding, chunk_embedding)

            scored.append({
                'index': i,
                'text': chunk,
                'score': similarity
            })

        # Sort by similarity descending
        scored.sort(key=lambda x: x['score'], reverse=True)

        return scored

    def _format_extraction_hints(self, similar_docs: List[Dict]) -> str:
        """
        Format extraction hints from similar documents.

        Provides examples of what was extracted from similar documents
        to help guide the current extraction.
        """
        if not similar_docs:
            return ""

        hints = []
        for doc in similar_docs[:2]:  # Top 2 similar docs
            similarity = doc.get('similarity', 0)
            extracted_values = doc.get('extracted_values', {})

            if extracted_values:
                hint_parts = [f"Similar document (similarity: {similarity:.2f}):"]
                for field, value in list(extracted_values.items())[:5]:
                    hint_parts.append(f"  - {field}: {value}")
                hints.append("\n".join(hint_parts))

        return "\n\n".join(hints)

    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Tries to split at natural boundaries (paragraphs, sentences).
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to find a good break point
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break + 2

                # Otherwise look for sentence break
                elif '.' in text[start:end]:
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def _score_chunks(
        self,
        chunks: List[str],
        extraction_prompt: str = None
    ) -> List[Dict[str, Any]]:
        """
        Score chunks for relevance to extraction task.

        Uses keyword matching and medical document patterns.
        """
        # Medical keywords that indicate important content
        medical_keywords = [
            # Lab-related
            'result', 'value', 'range', 'reference', 'normal', 'abnormal',
            'high', 'low', 'critical', 'flag', 'unit', 'specimen',
            # Common tests
            'hemoglobin', 'glucose', 'creatinine', 'cholesterol', 'sodium',
            'potassium', 'wbc', 'rbc', 'platelet', 'hgb', 'hct',
            # Medication-related
            'medication', 'drug', 'dose', 'dosage', 'mg', 'ml', 'tablet',
            'capsule', 'prescription', 'refill', 'sig', 'dispense',
            # Clinical-related
            'diagnosis', 'impression', 'finding', 'assessment', 'plan',
            'history', 'symptoms', 'condition', 'treatment',
            # Patient-related
            'patient', 'name', 'dob', 'date', 'provider', 'physician'
        ]

        # Extract keywords from extraction prompt if provided
        prompt_keywords = []
        if extraction_prompt:
            # Simple keyword extraction
            words = re.findall(r'\b[a-z]+\b', extraction_prompt.lower())
            prompt_keywords = [w for w in words if len(w) > 3]

        scored = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = 0.0

            # Score based on medical keywords
            for keyword in medical_keywords:
                if keyword in chunk_lower:
                    score += 0.1

            # Score based on prompt keywords
            for keyword in prompt_keywords:
                if keyword in chunk_lower:
                    score += 0.2

            # Bonus for numeric content (likely test results)
            numeric_matches = len(re.findall(r'\d+\.?\d*', chunk))
            score += min(numeric_matches * 0.05, 0.5)

            # Bonus for reference range patterns
            if re.search(r'\d+\s*-\s*\d+', chunk):
                score += 0.3

            # Penalty for very short chunks
            if len(chunk) < 100:
                score *= 0.5

            scored.append({
                'index': i,
                'text': chunk,
                'score': min(score, 1.0)
            })

        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)

        return scored

    def _select_chunks(
        self,
        scored_chunks: List[Dict[str, Any]],
        max_length: int
    ) -> List[Dict[str, Any]]:
        """
        Select top chunks that fit within max_length.

        Maintains some order awareness by including neighboring chunks.
        """
        selected = []
        total_length = 0

        for chunk in scored_chunks:
            chunk_length = len(chunk['text'])

            if total_length + chunk_length <= max_length:
                selected.append(chunk)
                total_length += chunk_length
            else:
                # Try to fit at least one chunk
                if not selected:
                    # Truncate the chunk to fit
                    truncated_text = chunk['text'][:max_length]
                    selected.append({
                        'index': chunk['index'],
                        'text': truncated_text,
                        'score': chunk['score']
                    })
                break

        # Sort by original position for coherent reading
        selected.sort(key=lambda x: x['index'])

        return selected

    def _analyze_document(self, ctx: RetrievalContext) -> Dict[str, Any]:
        """
        Analyze document characteristics for strategy selection.
        """
        text = ctx.text
        text_lower = text.lower()

        # Check length
        is_short = len(text) <= self.max_context_length

        # Count sections
        section_patterns = [
            r'^[A-Z][A-Z\s]+:',  # "CHEMISTRY:"
            r'^#{1,3}\s',        # Markdown headers
            r'^[A-Z][A-Z\s]+$'   # Uppercase line (section header)
        ]
        section_count = 0
        for line in text.split('\n'):
            for pattern in section_patterns:
                if re.match(pattern, line.strip()):
                    section_count += 1
                    break

        has_clear_sections = section_count >= 3

        # Check complexity
        # Complex = long + many sections + multiple content types
        has_tables = '|' in text or '\t' in text
        has_lists = bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE))
        has_multiple_pages = '--- Page' in text or len(text) > 10000

        is_complex = (
            has_multiple_pages and
            (has_tables or has_lists) and
            section_count > 5
        )

        # Check if layout info indicates tables
        if ctx.layout:
            if hasattr(ctx.layout, 'has_tables') and ctx.layout.has_tables:
                has_tables = True
            if hasattr(ctx.layout, 'has_sections') and ctx.layout.has_sections:
                has_clear_sections = True

        return {
            'is_short': is_short,
            'has_clear_sections': has_clear_sections,
            'is_complex': is_complex,
            'section_count': section_count,
            'has_tables': has_tables,
            'has_lists': has_lists,
            'has_multiple_pages': has_multiple_pages,
            'text_length': len(text)
        }

    async def store_extraction(
        self,
        text: str,
        extracted_values: Dict[str, Any],
        template_id: Optional[str] = None,
        source_file: str = "",
        confidence: float = 0.0
    ) -> Optional[str]:
        """
        Store successful extraction in vector store for future reference.

        This is key to Unstract's learning approach:
        - Store embeddings of documents with successful extractions
        - Future similar documents can use these as hints
        - Improves extraction accuracy over time

        Args:
            text: Document text (used for embedding)
            extracted_values: Successfully extracted field values
            template_id: Optional template identifier
            source_file: Source file path
            confidence: Extraction confidence score

        Returns:
            Document ID if stored, None if storage failed
        """
        if not self.vector_store:
            logger.debug("Vector store not available, skipping storage")
            return None

        # Only store high-confidence extractions
        min_store_confidence = self.config.get('min_store_confidence', 0.7)
        if confidence < min_store_confidence:
            logger.debug(f"Confidence {confidence} below threshold {min_store_confidence}, skipping storage")
            return None

        try:
            doc_id = await self.vector_store.store(
                text=text[:2000],  # Use first 2000 chars for embedding
                extracted_values=extracted_values,
                template_id=template_id,
                source_file=source_file,
                confidence=confidence
            )
            logger.info(f"Stored extraction to vector store: {doc_id}")
            return doc_id
        except Exception as e:
            logger.warning(f"Failed to store extraction: {e}")
            return None


# Convenience functions
async def retrieve_context(
    text: str,
    strategy: str = "router",
    extraction_prompt: str = None,
    config: Dict[str, Any] = None
) -> RetrievalResult:
    """
    Convenience function for adaptive retrieval.

    Args:
        text: Document text
        strategy: Strategy name ("simple", "router", "fusion", "chunked")
        extraction_prompt: The extraction prompt
        config: Optional configuration

    Returns:
        RetrievalResult with context for LLM
    """
    retrieval = AdaptiveRetrieval(config or {})
    strategy_enum = RetrievalStrategy(strategy)
    return await retrieval.retrieve(
        text=text,
        strategy=strategy_enum,
        extraction_prompt=extraction_prompt
    )
