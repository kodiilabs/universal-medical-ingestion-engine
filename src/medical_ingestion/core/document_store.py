# ============================================================================
# src/medical_ingestion/core/document_store.py
# ============================================================================
"""
Document Store

Persists processed document results to SQLite so they survive API restarts.
Follows the same pattern as audit.py — raw sqlite3, JSON for complex fields.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Default location alongside other data DBs
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "documents.db"


class DocumentStore:
    """
    SQLite-backed store for processed document results.

    Stores the full job dict (same shape the frontend expects from /api/v2/jobs)
    so the API can serve it back without any transformation.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_database()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_database(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                job_id          TEXT PRIMARY KEY,
                file_name       TEXT NOT NULL,
                file_path       TEXT,
                document_type   TEXT,
                status          TEXT NOT NULL DEFAULT 'completed',
                confidence      REAL,
                requires_review INTEGER DEFAULT 0,
                created_at      TEXT NOT NULL,
                completed_at    TEXT,
                total_time      REAL,
                -- Full job payload as JSON (what the frontend expects)
                job_data        TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_status
            ON documents (status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_type
            ON documents (document_type)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_created
            ON documents (created_at DESC)
        """)

        conn.commit()
        conn.close()
        logger.info(f"Document store initialized: {self.db_path}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def save(self, job: Dict[str, Any]) -> None:
        """
        Persist a completed (or failed) job.

        Args:
            job: The full job dict from v2_processing_jobs.
        """
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        cur.execute("""
            INSERT OR REPLACE INTO documents
                (job_id, file_name, file_path, document_type, status,
                 confidence, requires_review, created_at, completed_at,
                 total_time, job_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.get("job_id"),
            job.get("file_name", ""),
            job.get("file_path", ""),
            job.get("document_type", "unknown"),
            job.get("status", "completed"),
            job.get("confidence"),
            1 if job.get("requires_review") else 0,
            job.get("created_at", datetime.now().isoformat()),
            job.get("completed_at"),
            job.get("total_time"),
            json.dumps(job, default=str),
        ))

        conn.commit()
        conn.close()
        logger.info(f"Saved document {job.get('job_id')} to store")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single document by job_id."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("SELECT job_data FROM documents WHERE job_id = ?", (job_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None

    def list_all(
        self,
        status: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List stored documents, newest first.

        Args:
            status: Filter by status (completed, failed, etc.)
            document_type: Filter by document type
            limit: Max rows to return
            offset: Pagination offset
        """
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        query = "SELECT job_data FROM documents WHERE 1=1"
        params: list = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if document_type:
            query += " AND document_type = ?"
            params.append(document_type)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        return [json.loads(r[0]) for r in rows]

    def count(self, status: Optional[str] = None) -> int:
        """Count documents, optionally filtered by status."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        if status:
            cur.execute("SELECT COUNT(*) FROM documents WHERE status = ?", (status,))
        else:
            cur.execute("SELECT COUNT(*) FROM documents")
        n = cur.fetchone()[0]
        conn.close()
        return n

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete(self, job_id: str) -> bool:
        """Delete a document by job_id. Returns True if a row was removed."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE job_id = ?", (job_id,))
        deleted = cur.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
