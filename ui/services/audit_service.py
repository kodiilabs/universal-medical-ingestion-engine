# ============================================================================
# ui/services/audit_service.py
# ============================================================================
"""
Audit Service

Interfaces with the audit system for logging and retrieval.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging
import sqlite3
import json

logger = logging.getLogger(__name__)


class AuditService:
    """
    Service layer for audit trail management.

    Wraps the AuditLogger for use in Streamlit UI.
    """

    _instance = None
    _audit_logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._audit_logger is None:
            self._init_audit_logger()

    def _init_audit_logger(self):
        """Initialize the audit logger."""
        from src.medical_ingestion.core.audit import AuditLogger
        self._audit_logger = AuditLogger()
        logger.info("Audit logger initialized")

    @property
    def is_available(self) -> bool:
        """Check if audit system is available."""
        return self._audit_logger is not None

    def log_event(
        self,
        action: str,
        details: str = "",
        document_id: Optional[str] = None,
        status: str = "success",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log an audit event.

        Args:
            action: Event action type
            details: Event details
            document_id: Associated document ID
            status: Event status (success, warning, error, critical)
            metadata: Additional metadata
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
            'document_id': document_id,
            'status': status,
            'user': 'system',
            'metadata': metadata or {}
        }

        # Direct database insert for UI events
        conn = sqlite3.connect(str(self._audit_logger.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO processing_events (
                document_id, timestamp, event_type, metadata
            ) VALUES (?, ?, ?, ?)
        """, (
            document_id or 'ui_event',
            datetime.now(),
            action,
            json.dumps(event)
        ))

        conn.commit()
        conn.close()

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent audit events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of audit events
        """
        conn = sqlite3.connect(str(self._audit_logger.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM processing_events
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        events = []
        for row in cursor.fetchall():
            event = dict(row)
            # Parse metadata JSON if present
            if event.get('metadata'):
                try:
                    event['metadata'] = json.loads(event['metadata'])
                except:
                    pass
            events.append(event)

        conn.close()
        return events

    def get_document_trail(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get complete audit trail for a document.

        Args:
            document_id: Document ID

        Returns:
            List of audit trail entries
        """
        return self._audit_logger.get_trail(document_id)

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dict with processing stats
        """
        conn = sqlite3.connect(str(self._audit_logger.db_path))
        cursor = conn.cursor()

        # Total documents
        cursor.execute("SELECT COUNT(*) FROM processing_events WHERE event_type = 'processing_complete'")
        total_processed = cursor.fetchone()[0]

        # Success rate
        cursor.execute("SELECT COUNT(*) FROM processing_events WHERE success = 1")
        successes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM processing_events WHERE success = 0")
        failures = cursor.fetchone()[0]

        # Average confidence
        cursor.execute("SELECT AVG(overall_confidence) FROM processing_events WHERE overall_confidence IS NOT NULL")
        avg_confidence = cursor.fetchone()[0] or 0.0

        # Documents requiring review
        cursor.execute("SELECT COUNT(*) FROM processing_events WHERE requires_review = 1")
        needs_review = cursor.fetchone()[0]

        # Document types breakdown
        cursor.execute("""
            SELECT document_type, COUNT(*) as count
            FROM processing_events
            WHERE document_type IS NOT NULL
            GROUP BY document_type
        """)
        type_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            'total_processed': total_processed,
            'successes': successes,
            'failures': failures,
            'success_rate': successes / (successes + failures) if (successes + failures) > 0 else 0,
            'avg_confidence': avg_confidence,
            'needs_review': needs_review,
            'type_breakdown': type_breakdown
        }

    def clear_audit_log(self) -> bool:
        """
        Clear all audit log entries.

        Returns:
            True if successful
        """
        conn = sqlite3.connect(str(self._audit_logger.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM processing_events")
        cursor.execute("DELETE FROM agent_executions")

        conn.commit()
        conn.close()

        return True

    def export_audit_log(self, format: str = 'json') -> str:
        """
        Export audit log.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Exported data as string
        """
        events = self.get_recent_events(limit=10000)

        if format == 'csv':
            lines = ['timestamp,action,document_id,status,details']
            for event in events:
                lines.append(
                    f"{event.get('timestamp', '')},{event.get('event_type', '')},"
                    f"{event.get('document_id', '')},{event.get('success', '')},"
                    f"\"{event.get('error', '')}\""
                )
            return '\n'.join(lines)
        else:
            return json.dumps(events, indent=2, default=str)
