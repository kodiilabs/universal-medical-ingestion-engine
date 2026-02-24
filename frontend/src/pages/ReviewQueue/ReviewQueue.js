// ============================================================================
// Review Queue Page - Documents requiring human review
// ============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  AlertTriangle,
  Clock,
  FileText,
  Loader2,
  CheckCircle,
  XCircle,
  Eye
} from 'lucide-react';
import { listJobs } from '../../services/api';
import { ConfidenceBadge } from '../../components/shared';
import './ReviewQueue.css';

const PRIORITY_CONFIG = {
  CRITICAL: { icon: AlertCircle, color: '#EF4444', label: 'Critical' },
  HIGH: { icon: AlertTriangle, color: '#F97316', label: 'High' },
  MEDIUM: { icon: Clock, color: '#F59E0B', label: 'Medium' },
  LOW: { icon: Clock, color: '#6B7280', label: 'Low' },
};

const ReviewQueue = () => {
  const navigate = useNavigate();

  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [priorityFilter, setPriorityFilter] = useState('');

  // Fetch documents needing review
  const fetchDocuments = useCallback(async () => {
    try {
      const response = await listJobs();
      // Filter to only documents requiring review
      const reviewDocs = (response.jobs || []).filter(
        (doc) => doc.result?.requires_review || doc.status === 'failed'
      );
      setDocuments(reviewDocs);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Get priority from document
  const getPriority = (doc) => {
    if (doc.status === 'failed') return 'CRITICAL';
    if (doc.result?.confidence < 0.5) return 'CRITICAL';
    if (doc.result?.confidence < 0.7) return 'HIGH';
    if (doc.result?.critical_findings?.length > 0) return 'CRITICAL';
    return 'MEDIUM';
  };

  // Get review reasons
  const getReviewReasons = (doc) => {
    const reasons = [];
    if (doc.status === 'failed') reasons.push('Processing failed');
    if (doc.result?.confidence < 0.7) reasons.push('Low confidence');
    if (doc.result?.critical_findings?.length > 0) reasons.push('Critical findings');
    if (doc.result?.review_reasons) reasons.push(...doc.result.review_reasons);
    return reasons.length > 0 ? reasons : ['Manual review requested'];
  };

  // Filter documents
  const filteredDocuments = documents.filter((doc) => {
    if (!priorityFilter) return true;
    return getPriority(doc) === priorityFilter;
  });

  // Sort by priority
  const priorityOrder = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
  const sortedDocuments = [...filteredDocuments].sort(
    (a, b) => priorityOrder[getPriority(a)] - priorityOrder[getPriority(b)]
  );

  const handleReview = (doc) => {
    navigate(`/documents/${doc.job_id}`);
  };

  if (loading) {
    return (
      <div className="review-loading">
        <Loader2 size={32} className="spinning" />
        <p>Loading review queue...</p>
      </div>
    );
  }

  return (
    <div className="review-queue-page">
      <div className="review-header">
        <div className="header-left">
          <h1>Review Queue</h1>
          <span className="queue-count">{documents.length} documents pending review</span>
        </div>
        <div className="header-right">
          <div className="priority-filter">
            <span className="filter-label">Priority:</span>
            <select
              value={priorityFilter}
              onChange={(e) => setPriorityFilter(e.target.value)}
              className="input"
            >
              <option value="">All Priorities</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MEDIUM">Medium</option>
              <option value="LOW">Low</option>
            </select>
          </div>
        </div>
      </div>

      {/* Priority Stats */}
      <div className="priority-stats">
        {Object.entries(PRIORITY_CONFIG).map(([key, config]) => {
          const count = documents.filter((d) => getPriority(d) === key).length;
          const Icon = config.icon;
          return (
            <div
              key={key}
              className={`stat-card ${priorityFilter === key ? 'active' : ''}`}
              onClick={() => setPriorityFilter(priorityFilter === key ? '' : key)}
            >
              <div className="stat-icon" style={{ color: config.color }}>
                <Icon size={20} />
              </div>
              <div className="stat-content">
                <span className="stat-count">{count}</span>
                <span className="stat-label">{config.label}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Review Queue */}
      <div className="review-list">
        {sortedDocuments.length === 0 ? (
          <div className="empty-state">
            <CheckCircle size={48} />
            <h3>No documents pending review</h3>
            <p>All documents have been reviewed or processed successfully</p>
          </div>
        ) : (
          sortedDocuments.map((doc) => {
            const priority = getPriority(doc);
            const config = PRIORITY_CONFIG[priority];
            const Icon = config.icon;
            const reasons = getReviewReasons(doc);

            return (
              <div key={doc.job_id} className={`review-card ${priority.toLowerCase()}`}>
                <div className="review-priority">
                  <Icon size={24} style={{ color: config.color }} />
                </div>

                <div className="review-content">
                  <div className="review-header-row">
                    <div className="review-file">
                      <FileText size={18} />
                      <span className="file-name">{doc.file_name}</span>
                    </div>
                    <span className={`priority-badge ${priority.toLowerCase()}`}>
                      {config.label}
                    </span>
                  </div>

                  <div className="review-meta">
                    <span className="doc-type">{doc.document_type || 'Unknown'}</span>
                    <span className="separator">•</span>
                    <span className="doc-date">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </span>
                    {doc.result?.confidence && (
                      <>
                        <span className="separator">•</span>
                        <ConfidenceBadge score={doc.result.confidence} showLabel={false} size="sm" />
                      </>
                    )}
                  </div>

                  <div className="review-reasons">
                    {reasons.map((reason, index) => (
                      <span key={index} className="reason-tag">
                        {reason}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="review-actions">
                  <button
                    className="btn btn-primary"
                    onClick={() => handleReview(doc)}
                  >
                    <Eye size={16} />
                    Review
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default ReviewQueue;
