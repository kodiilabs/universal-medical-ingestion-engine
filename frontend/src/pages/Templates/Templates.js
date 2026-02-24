// ============================================================================
// Template Catalog Page - Admin view of document templates
// ============================================================================

import React, { useState } from 'react';
import {
  FileText,
  Plus,
  Settings,
  CheckCircle,
  Clock,
  Edit2,
  Trash2,
  Copy,
  Eye
} from 'lucide-react';
import './Templates.css';

// Mock template data - in production this would come from API
const MOCK_TEMPLATES = [
  {
    id: '1',
    name: 'Quest Diagnostics CBC',
    document_type: 'lab',
    status: 'active',
    fields: 15,
    accuracy: 0.96,
    uses: 234,
    last_updated: '2024-01-10',
  },
  {
    id: '2',
    name: 'LabCorp Metabolic Panel',
    document_type: 'lab',
    status: 'active',
    fields: 22,
    accuracy: 0.94,
    uses: 187,
    last_updated: '2024-01-08',
  },
  {
    id: '3',
    name: 'Radiology Report - Standard',
    document_type: 'radiology',
    status: 'active',
    fields: 8,
    accuracy: 0.91,
    uses: 156,
    last_updated: '2024-01-05',
  },
  {
    id: '4',
    name: 'Prescription - Generic',
    document_type: 'prescription',
    status: 'active',
    fields: 12,
    accuracy: 0.89,
    uses: 98,
    last_updated: '2024-01-03',
  },
  {
    id: '5',
    name: 'Unknown Format - Draft',
    document_type: 'unknown',
    status: 'draft',
    fields: 5,
    accuracy: 0.72,
    uses: 12,
    last_updated: '2024-01-12',
  },
];

const Templates = () => {
  const [templates] = useState(MOCK_TEMPLATES);
  const [typeFilter, setTypeFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState(null);

  // Filter templates
  const filteredTemplates = templates.filter((template) => {
    const matchesType = !typeFilter || template.document_type === typeFilter;
    const matchesStatus = !statusFilter || template.status === statusFilter;
    return matchesType && matchesStatus;
  });

  // Stats
  const stats = {
    total: templates.length,
    active: templates.filter((t) => t.status === 'active').length,
    draft: templates.filter((t) => t.status === 'draft').length,
    avgAccuracy: (
      templates.reduce((sum, t) => sum + t.accuracy, 0) / templates.length
    ).toFixed(2),
  };

  return (
    <div className="templates-page">
      <div className="templates-header">
        <div className="header-left">
          <h1>Template Catalog</h1>
          <p>Manage document extraction templates</p>
        </div>
        <div className="header-right">
          <button className="btn btn-primary">
            <Plus size={18} />
            Create Template
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="template-stats">
        <div className="stat-card">
          <FileText size={24} />
          <div className="stat-info">
            <span className="stat-value">{stats.total}</span>
            <span className="stat-label">Total Templates</span>
          </div>
        </div>
        <div className="stat-card">
          <CheckCircle size={24} />
          <div className="stat-info">
            <span className="stat-value">{stats.active}</span>
            <span className="stat-label">Active</span>
          </div>
        </div>
        <div className="stat-card">
          <Clock size={24} />
          <div className="stat-info">
            <span className="stat-value">{stats.draft}</span>
            <span className="stat-label">Drafts</span>
          </div>
        </div>
        <div className="stat-card">
          <Settings size={24} />
          <div className="stat-info">
            <span className="stat-value">{Math.round(stats.avgAccuracy * 100)}%</span>
            <span className="stat-label">Avg Accuracy</span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="templates-toolbar">
        <div className="filter-group">
          <label>Document Type</label>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="input"
          >
            <option value="">All Types</option>
            <option value="lab">Lab Report</option>
            <option value="radiology">Radiology</option>
            <option value="prescription">Prescription</option>
            <option value="pathology">Pathology</option>
            <option value="unknown">Unknown</option>
          </select>
        </div>
        <div className="filter-group">
          <label>Status</label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="input"
          >
            <option value="">All Statuses</option>
            <option value="active">Active</option>
            <option value="draft">Draft</option>
            <option value="archived">Archived</option>
          </select>
        </div>
      </div>

      {/* Templates Grid */}
      <div className="templates-grid">
        {filteredTemplates.map((template) => (
          <div
            key={template.id}
            className={`template-card ${selectedTemplate === template.id ? 'selected' : ''}`}
            onClick={() => setSelectedTemplate(template.id)}
          >
            <div className="template-header">
              <div className="template-icon">
                <FileText size={20} />
              </div>
              <span className={`status-badge ${template.status}`}>
                {template.status}
              </span>
            </div>

            <h3 className="template-name">{template.name}</h3>

            <div className="template-type">
              {template.document_type}
            </div>

            <div className="template-stats-row">
              <div className="template-stat">
                <span className="stat-num">{template.fields}</span>
                <span className="stat-text">Fields</span>
              </div>
              <div className="template-stat">
                <span className="stat-num">{Math.round(template.accuracy * 100)}%</span>
                <span className="stat-text">Accuracy</span>
              </div>
              <div className="template-stat">
                <span className="stat-num">{template.uses}</span>
                <span className="stat-text">Uses</span>
              </div>
            </div>

            <div className="template-footer">
              <span className="last-updated">
                Updated {new Date(template.last_updated).toLocaleDateString()}
              </span>
            </div>

            <div className="template-actions">
              <button className="action-btn" title="View">
                <Eye size={16} />
              </button>
              <button className="action-btn" title="Edit">
                <Edit2 size={16} />
              </button>
              <button className="action-btn" title="Duplicate">
                <Copy size={16} />
              </button>
              <button className="action-btn danger" title="Delete">
                <Trash2 size={16} />
              </button>
            </div>
          </div>
        ))}

        {/* Add New Template Card */}
        <div className="template-card add-new">
          <div className="add-content">
            <Plus size={32} />
            <span>Create New Template</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Templates;
