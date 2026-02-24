// ============================================================================
// Documents List Page - Table with filters, selection, and delete
// ============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  Filter,
  FileText,
  Loader2,
  ChevronDown,
  RefreshCw,
  Trash2,
  CheckSquare,
  Square,
  X
} from 'lucide-react';
import { listJobs, deleteJob, deleteJobsBatch } from '../../services/api';
import { StatusBadge, ConfidenceBadge } from '../../components/shared';
import './Documents.css';

const DOCUMENT_TYPES = [
  {
    value: '',
    label: 'All Types',
    icon: 'Files',
    color: '#6B7280',
    category: null,
    priority: 0
  },
  {
    value: 'lab',
    label: 'Lab Report',
    icon: 'TestTube',
    color: '#3B82F6',
    category: 'diagnostic',
    priority: 1
  },
  {
    value: 'radiology',
    label: 'Radiology',
    icon: 'Scan',
    color: '#8B5CF6',
    category: 'diagnostic',
    priority: 2
  },
  {
    value: 'pathology',
    label: 'Pathology',
    icon: 'Microscope',
    color: '#EC4899',
    category: 'diagnostic',
    priority: 3
  },
  {
    value: 'prescription',
    label: 'Prescription',
    icon: 'Pill',
    color: '#10B981',
    category: 'treatment',
    priority: 4
  },
  {
    value: 'discharge_summary',
    label: 'Discharge Summary',
    icon: 'FileCheck',
    color: '#06B6D4',
    category: 'clinical',
    priority: 5
  },
  {
    value: 'consultation',
    label: 'Consultation Note',
    icon: 'UserRound',
    color: '#8B5CF6',
    category: 'clinical',
    priority: 6
  },
  {
    value: 'progress_note',
    label: 'Progress Note',
    icon: 'ClipboardList',
    color: '#0EA5E9',
    category: 'clinical',
    priority: 7
  },
  {
    value: 'operative_report',
    label: 'Operative Report',
    icon: 'Scissors',
    color: '#EF4444',
    category: 'clinical',
    priority: 8
  },
  {
    value: 'immunization',
    label: 'Immunization Record',
    icon: 'Syringe',
    color: '#10B981',
    category: 'preventive',
    priority: 9
  },
  {
    value: 'referral',
    label: 'Referral',
    icon: 'ArrowRightLeft',
    color: '#F59E0B',
    category: 'administrative',
    priority: 10
  },
  {
    value: 'insurance',
    label: 'Insurance Document',
    icon: 'ShieldCheck',
    color: '#6366F1',
    category: 'administrative',
    priority: 11
  },
  {
    value: 'consent_form',
    label: 'Consent Form',
    icon: 'FileSignature',
    color: '#8B5CF6',
    category: 'administrative',
    priority: 12
  },
  {
    value: 'vital_signs',
    label: 'Vital Signs',
    icon: 'Activity',
    color: '#F43F5E',
    category: 'clinical',
    priority: 13
  },
  {
    value: 'allergy_record',
    label: 'Allergy Record',
    icon: 'AlertTriangle',
    color: '#F59E0B',
    category: 'clinical',
    priority: 14
  },
  {
    value: 'medication_list',
    label: 'Medication List',
    icon: 'Tablets',
    color: '#14B8A6',
    category: 'treatment',
    priority: 15
  },
  {
    value: 'billing',
    label: 'Billing Statement',
    icon: 'Receipt',
    color: '#84CC16',
    category: 'administrative',
    priority: 16
  },
  {
    value: 'advance_directive',
    label: 'Advance Directive',
    icon: 'ScrollText',
    color: '#A855F7',
    category: 'administrative',
    priority: 17
  },
  {
    value: 'unknown',
    label: 'Unknown',
    icon: 'FileQuestion',
    color: '#9CA3AF',
    category: null,
    priority: 99
  },
];

const STATUS_OPTIONS = [
  { value: '', label: 'All Statuses' },
  { value: 'pending', label: 'Pending' },
  { value: 'processing', label: 'Processing' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
];

const Documents = () => {
  const navigate = useNavigate();

  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // Selection & delete
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [deleting, setDeleting] = useState(false);

  // Fetch documents
  const fetchDocuments = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true);

    try {
      const response = await listJobs();
      setDocuments(response.jobs || []);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Filter documents
  const filteredDocuments = documents.filter((doc) => {
    const matchesSearch = searchQuery === '' ||
      doc.file_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.document_type?.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesType = typeFilter === '' || doc.document_type === typeFilter;
    const matchesStatus = statusFilter === '' || doc.status === statusFilter;

    return matchesSearch && matchesType && matchesStatus;
  });

  const handleRowClick = (doc) => {
    if (selectionMode) {
      toggleSelect(doc.job_id);
      return;
    }
    if (doc.status === 'completed' || doc.status === 'failed') {
      navigate(`/documents/${doc.job_id}`);
    } else {
      navigate(`/dashboard?job=${doc.job_id}`);
    }
  };

  // Selection helpers
  const toggleSelect = (jobId) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(jobId)) next.delete(jobId);
      else next.add(jobId);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === filteredDocuments.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredDocuments.map(d => d.job_id)));
    }
  };

  const exitSelectionMode = () => {
    setSelectionMode(false);
    setSelectedIds(new Set());
  };

  // Delete single document
  const handleDeleteSingle = async (e, jobId) => {
    e.stopPropagation();
    if (!window.confirm('Delete this document? This cannot be undone.')) return;

    try {
      setDeleting(true);
      await deleteJob(jobId);
      setDocuments(prev => prev.filter(d => d.job_id !== jobId));
      setSelectedIds(prev => {
        const next = new Set(prev);
        next.delete(jobId);
        return next;
      });
    } catch (err) {
      console.error('Failed to delete document:', err);
      alert('Failed to delete document. Please try again.');
    } finally {
      setDeleting(false);
    }
  };

  // Batch delete
  const handleDeleteSelected = async () => {
    const count = selectedIds.size;
    if (count === 0) return;
    if (!window.confirm(`Delete ${count} document${count > 1 ? 's' : ''}? This cannot be undone.`)) return;

    try {
      setDeleting(true);
      await deleteJobsBatch([...selectedIds]);
      setDocuments(prev => prev.filter(d => !selectedIds.has(d.job_id)));
      exitSelectionMode();
    } catch (err) {
      console.error('Failed to delete documents:', err);
      alert('Failed to delete some documents. Please try again.');
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="documents-loading">
        <Loader2 size={32} className="spinning" />
        <p>Loading documents...</p>
      </div>
    );
  }

  return (
    <div className="documents-page">
      <div className="documents-header">
        <div className="header-left">
          <h1>Documents</h1>
          <span className="doc-count">{filteredDocuments.length} documents</span>
        </div>
        <div className="header-right">
          {selectionMode ? (
            <>
              <span className="selection-count">{selectedIds.size} selected</span>
              <button
                className="btn btn-danger"
                onClick={handleDeleteSelected}
                disabled={selectedIds.size === 0 || deleting}
              >
                {deleting ? <Loader2 size={16} className="spinning" /> : <Trash2 size={16} />}
                Delete Selected
              </button>
              <button className="btn btn-ghost" onClick={exitSelectionMode}>
                <X size={16} />
                Cancel
              </button>
            </>
          ) : (
            <>
              <button
                className="btn btn-ghost"
                onClick={() => fetchDocuments(true)}
                disabled={refreshing}
              >
                <RefreshCw size={18} className={refreshing ? 'spinning' : ''} />
              </button>
              <button
                className="btn btn-ghost"
                onClick={() => setSelectionMode(true)}
              >
                <CheckSquare size={18} />
                Select
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => setShowFilters(!showFilters)}
              >
                <Filter size={18} />
                Filters
                <ChevronDown size={16} className={showFilters ? 'rotated' : ''} />
              </button>
              <button
                className="btn btn-primary"
                onClick={() => navigate('/')}
              >
                Upload New
              </button>
            </>
          )}
        </div>
      </div>

      {/* Search & Filters */}
      <div className="documents-toolbar">
        <div className="search-box">
          <Search size={18} />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>

        {showFilters && (
          <div className="filters-row">
            <div className="filter-group">
              <label>Document Type</label>
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="input"
              >
                {DOCUMENT_TYPES.map(({ value, label }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
            <div className="filter-group">
              <label>Status</label>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="input"
              >
                {STATUS_OPTIONS.map(({ value, label }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
            <button
              className="btn btn-ghost clear-filters"
              onClick={() => {
                setSearchQuery('');
                setTypeFilter('');
                setStatusFilter('');
              }}
            >
              Clear Filters
            </button>
          </div>
        )}
      </div>

      {/* Documents Table */}
      <div className="documents-table-container">
        {filteredDocuments.length === 0 ? (
          <div className="empty-state">
            <FileText size={48} />
            <h3>No documents found</h3>
            <p>
              {documents.length === 0
                ? 'Upload a document to get started'
                : 'Try adjusting your filters'}
            </p>
          </div>
        ) : (
          <table className="table documents-table">
            <thead>
              <tr>
                {selectionMode && (
                  <th className="checkbox-col">
                    <button
                      className="checkbox-btn"
                      onClick={toggleSelectAll}
                    >
                      {selectedIds.size === filteredDocuments.length
                        ? <CheckSquare size={18} />
                        : <Square size={18} />
                      }
                    </button>
                  </th>
                )}
                <th>File Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Created</th>
                <th>Review</th>
                {!selectionMode && <th className="actions-col"></th>}
              </tr>
            </thead>
            <tbody>
              {filteredDocuments.map((doc) => (
                <tr
                  key={doc.job_id}
                  className={`clickable ${selectedIds.has(doc.job_id) ? 'row-selected' : ''}`}
                  onClick={() => handleRowClick(doc)}
                >
                  {selectionMode && (
                    <td className="checkbox-col">
                      <span className="checkbox-btn">
                        {selectedIds.has(doc.job_id)
                          ? <CheckSquare size={18} />
                          : <Square size={18} />
                        }
                      </span>
                    </td>
                  )}
                  <td>
                    <div className="file-cell">
                      <FileText size={18} />
                      <span className="file-name">{doc.file_name}</span>
                    </div>
                  </td>
                  <td>
                    <span className="type-badge">{doc.document_type || 'auto'}</span>
                  </td>
                  <td>
                    <StatusBadge status={doc.status} />
                  </td>
                  <td>
                    {doc.result?.confidence ? (
                      <ConfidenceBadge score={doc.result.confidence} showLabel={false} />
                    ) : (
                      <span className="no-data">-</span>
                    )}
                  </td>
                  <td>
                    <span className="date-cell">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </span>
                  </td>
                  <td>
                    {doc.result?.requires_review ? (
                      <span className="review-required">Required</span>
                    ) : doc.status === 'completed' ? (
                      <span className="no-review">-</span>
                    ) : (
                      <span className="no-data">-</span>
                    )}
                  </td>
                  {!selectionMode && (
                    <td className="actions-col">
                      <button
                        className="row-delete-btn"
                        onClick={(e) => handleDeleteSingle(e, doc.job_id)}
                        title="Delete document"
                        disabled={deleting}
                      >
                        <Trash2 size={15} />
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default Documents;
