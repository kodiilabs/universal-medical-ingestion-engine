# ============================================================================
# src/medical_ingestion/core/audit.py
# ============================================================================
"""
Audit Trail Logger

HIPAA compliance requires complete audit trails of all processing.

This module logs:
- Every agent decision
- Every confidence score
- Every warning/flag
- Complete document lineage (PDF → agents → FHIR)

All audit data stored locally in SQLite database.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import sqlite3
from datetime import datetime
import json

from ..config.base_config import base_settings
from .context.processing_context import ProcessingContext


class AuditLogger:
    """
    Audit trail management for HIPAA compliance.
    
    Stores:
    - Document processing events
    - Agent executions and decisions
    - Confidence scores and warnings
    - Processing duration and errors
    
    All data persisted to local SQLite database.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or base_settings.AUDIT_DB_PATH
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create audit database schema if not exists"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Processing events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                document_path TEXT,
                document_type TEXT,
                processor TEXT,
                success BOOLEAN,
                error TEXT,
                processing_duration REAL,
                overall_confidence REAL,
                requires_review BOOLEAN,
                review_priority TEXT,
                metadata TEXT
            )
        """)
        
        # Agent executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                agent_name TEXT NOT NULL,
                decision TEXT,
                confidence REAL,
                reasoning TEXT,
                duration REAL,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES processing_events (document_id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_id 
            ON processing_events (document_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON processing_events (timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_document 
            ON agent_executions (document_id, agent_name)
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Audit database initialized: {self.db_path}")
    
    def log_processing_complete(self, context: ProcessingContext):
        """Log successful processing completion"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO processing_events (
                document_id, timestamp, event_type, document_path,
                document_type, processor, success, processing_duration,
                overall_confidence, requires_review, review_priority, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            context.document_id,
            datetime.now(),
            'processing_complete',
            str(context.document_path),
            context.document_type,
            context.template_id or 'unknown',
            True,
            context.processing_duration,
            context.overall_confidence,
            context.requires_review,
            context.review_priority.value if context.review_priority else None,
            json.dumps(context.get_summary())
        ))
        
        conn.commit()
        conn.close()
    
    def log_processing_error(self, context: ProcessingContext, error: str):
        """Log processing error"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO processing_events (
                document_id, timestamp, event_type, document_path,
                success, error, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            context.document_id,
            datetime.now(),
            'processing_error',
            str(context.document_path),
            False,
            error,
            json.dumps({"agent_executions": len(context.agent_executions)})
        ))
        
        conn.commit()
        conn.close()
    
    def get_trail(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve complete audit trail for a document"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all agent executions for this document
        cursor.execute("""
            SELECT * FROM agent_executions 
            WHERE document_id = ? 
            ORDER BY timestamp
        """, (document_id,))
        
        trail = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return trail