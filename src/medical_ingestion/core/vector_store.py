# ============================================================================
# src/medical_ingestion/core/vector_store.py
# ============================================================================
"""
Vector Store for Document Similarity

Inspired by Unstract's approach:
- Store embeddings of successfully processed documents
- Find similar documents to guide extraction on new PDFs
- Enables "near-miss" template matching for layout variations

Features:
- SQLite + numpy for lightweight vector storage
- Cosine similarity for document matching
- Stores extraction results with embeddings for transfer learning
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json
import hashlib
import sqlite3
import numpy as np
from datetime import datetime

from .config import get_config

# Singleton instance for shared vector store
_vector_store_instance: Optional['VectorStore'] = None


def get_vector_store(config: Dict[str, Any] = None) -> 'VectorStore':
    """Get shared VectorStore instance (singleton) - avoids reloading embedding model."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(config)
    return _vector_store_instance


@dataclass
class DocumentEmbedding:
    """Stored document with embedding and extraction results."""
    doc_id: str
    text_hash: str
    embedding: np.ndarray
    extracted_values: Dict[str, Any]
    template_id: Optional[str]
    source_file: str
    created_at: datetime
    confidence: float = 0.0


class VectorStore:
    """
    Lightweight vector store for document similarity.

    Uses sentence-transformers for embeddings and SQLite for storage.
    Designed to improve extraction accuracy on similar documents.
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Merge env config with passed config (passed config takes precedence)
        env_config = get_config()
        self.config = {**env_config, **(config or {})}
        self.logger = logging.getLogger(__name__)

        # Database path
        db_dir = Path(self.config.get('data_dir', 'data'))
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / 'vector_store.db'

        # Embedding model (lazy loaded)
        self._embedding_model = None
        self._embedding_dim = 384  # Default for all-MiniLM-L6-v2

        # Embedding cache (LRU-style, keyed by text hash)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 100

        # HTTP session for connection pooling (Ollama)
        self._http_session = None

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for vector storage."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text_hash TEXT UNIQUE,
                embedding BLOB,
                extracted_values TEXT,
                template_id TEXT,
                source_file TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_template_id ON documents(template_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_text_hash ON documents(text_hash)
        ''')

        conn.commit()
        conn.close()

        self.logger.debug(f"Vector store initialized at {self.db_path}")

    @property
    def embedding_backend(self) -> str:
        """Get embedding backend type."""
        return self.config.get('embedding_backend', 'ollama').lower()

    @property
    def embedding_model(self):
        """Lazy load embedding model (for sentence-transformers backend)."""
        if self._embedding_model is None:
            if self.embedding_backend == 'ollama':
                # Ollama doesn't need a loaded model - uses HTTP API
                self._embedding_model = "ollama"
                # nomic-embed-text-v2-moe:latest produces 768-dim embeddings
                self._embedding_dim = self.config.get('embedding_dim', 768)
                self.logger.info(f"Using Ollama embeddings: {self.config.get('embedding_model', 'nomic-embed-text-v2-moe:latest')}")
            elif self.embedding_backend == 'sentence_transformers':
                try:
                    from sentence_transformers import SentenceTransformer
                    model_name = self.config.get(
                        'embedding_model',
                        'all-MiniLM-L6-v2'  # Fast, good quality
                    )
                    self._embedding_model = SentenceTransformer(model_name)
                    self._embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
                    self.logger.info(f"Loaded embedding model: {model_name}")
                except ImportError:
                    self.logger.warning(
                        "sentence-transformers not installed. "
                        "Falling back to TF-IDF."
                    )
                    self._embedding_model = "tfidf_fallback"
            else:
                self._embedding_model = "tfidf_fallback"

        return self._embedding_model

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text with caching."""
        # Check cache first
        text_hash = self._text_hash(text)
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        # Compute embedding
        if self.embedding_model == "ollama":
            embedding = self._ollama_embedding(text)
        elif self.embedding_model == "tfidf_fallback":
            embedding = self._tfidf_embedding(text)
        else:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        # Cache result (simple LRU: remove oldest if full)
        if len(self._embedding_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        self._embedding_cache[text_hash] = embedding

        return embedding

    @property
    def http_session(self):
        """Lazy load HTTP session for connection pooling."""
        if self._http_session is None:
            import requests
            self._http_session = requests.Session()
        return self._http_session

    def _ollama_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API with connection pooling."""
        ollama_host = self.config.get('ollama_host', 'http://localhost:11434')
        model_name = self.config.get('embedding_model', 'nomic-embed-text-v2-moe:latest')

        try:
            response = self.http_session.post(
                f"{ollama_host}/api/embeddings",
                json={
                    "model": model_name,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json().get('embedding', [])
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Ollama embedding failed: {e}")
            return np.zeros(self._embedding_dim, dtype=np.float32)

    def _tfidf_embedding(self, text: str) -> np.ndarray:
        """Fallback TF-IDF based embedding."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Simple character n-grams for quick embedding
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=self._embedding_dim
        )

        try:
            # Fit on the text and transform
            vec = vectorizer.fit_transform([text]).toarray()[0]
            # Pad or truncate to embedding_dim
            if len(vec) < self._embedding_dim:
                vec = np.pad(vec, (0, self._embedding_dim - len(vec)))
            return vec[:self._embedding_dim]
        except Exception:
            return np.zeros(self._embedding_dim)

    def _text_hash(self, text: str) -> str:
        """Generate hash for text deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    async def store(
        self,
        text: str,
        extracted_values: Dict[str, Any],
        template_id: Optional[str] = None,
        source_file: str = "",
        confidence: float = 0.0
    ) -> str:
        """
        Store document embedding and extraction results.

        Args:
            text: Document text (first 2000 chars typically)
            extracted_values: Successfully extracted field values
            template_id: Template used (if any)
            source_file: Source file path
            confidence: Extraction confidence

        Returns:
            Document ID
        """
        text_hash = self._text_hash(text)

        # Check if already exists
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT doc_id FROM documents WHERE text_hash = ?",
            (text_hash,)
        )
        existing = cursor.fetchone()

        if existing:
            conn.close()
            self.logger.debug(f"Document already stored: {existing[0]}")
            return existing[0]

        # Compute embedding
        embedding = self._compute_embedding(text)

        # Generate doc_id
        import uuid
        doc_id = str(uuid.uuid4())[:12]

        # Store
        cursor.execute('''
            INSERT INTO documents (doc_id, text_hash, embedding, extracted_values,
                                   template_id, source_file, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_id,
            text_hash,
            embedding.tobytes(),
            json.dumps(extracted_values),
            template_id,
            source_file,
            confidence
        ))

        conn.commit()
        conn.close()

        self.logger.info(f"Stored document {doc_id} with {len(extracted_values)} values")
        return doc_id

    async def find_similar(
        self,
        text: str,
        top_k: int = 3,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Find similar documents to guide extraction.

        Args:
            text: Query text
            top_k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar documents with their extractions
        """
        # Compute query embedding
        query_embedding = self._compute_embedding(text)

        # Load all embeddings from DB
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT doc_id, embedding, extracted_values, template_id, confidence
            FROM documents
        ''')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate similarities
        results = []
        for doc_id, embedding_bytes, values_json, template_id, confidence in rows:
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            # Ensure same dimension
            if len(stored_embedding) != len(query_embedding):
                continue

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, stored_embedding)

            if similarity >= min_similarity:
                results.append({
                    'doc_id': doc_id,
                    'similarity': float(similarity),
                    'extracted_values': json.loads(values_json),
                    'template_id': template_id,
                    'extraction_confidence': confidence
                })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    async def get_by_template(
        self,
        template_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get documents that used a specific template."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT doc_id, extracted_values, confidence, created_at
            FROM documents
            WHERE template_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (template_id, limit))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'doc_id': doc_id,
                'extracted_values': json.loads(values_json),
                'confidence': confidence,
                'created_at': created_at
            }
            for doc_id, values_json, confidence, created_at in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]

        cursor.execute(
            "SELECT template_id, COUNT(*) FROM documents GROUP BY template_id"
        )
        by_template = dict(cursor.fetchall())

        cursor.execute("SELECT AVG(confidence) FROM documents")
        avg_confidence = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            'total_documents': total_docs,
            'by_template': by_template,
            'average_confidence': avg_confidence,
            'db_path': str(self.db_path)
        }

    def clear(self):
        """Clear all stored documents."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        conn.commit()
        conn.close()
        self.logger.info("Vector store cleared")
