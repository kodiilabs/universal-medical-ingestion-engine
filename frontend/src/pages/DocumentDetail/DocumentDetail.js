// ============================================================================
// Document Detail Page - Split view with PDF viewer and extracted data
// ============================================================================

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Document, Page, pdfjs } from 'react-pdf';
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Loader2,
  ArrowLeft,
  GitBranch,
  Download,
  AlertTriangle,
  RotateCcw,
  Trash2,
  ChevronDown,
  Send
} from 'lucide-react';
import { getJobStatus, getDocumentUrl, reprocessJob, deleteJob, sendChatMessage, getChatSuggestions, clearChatHistory } from '../../services/api';
import { StatusBadge, ConfidenceBadge } from '../../components/shared';
import './DocumentDetail.css';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

const TABS = [
  { id: 'structured', label: 'Structured Data' },
  { id: 'formatted', label: 'Formatted' },
  { id: 'json', label: 'JSON' },
  { id: 'fhir', label: 'FHIR' },
  { id: 'boxes', label: 'Bounding Boxes' },
  { id: 'chat', label: 'Chat' },
  { id: 'workflow', label: 'Workflow Log' },
];

const RECLASSIFY_TYPES = [
  { value: 'lab', label: 'Lab Report' },
  { value: 'radiology', label: 'Radiology' },
  { value: 'pathology', label: 'Pathology' },
  { value: 'prescription', label: 'Prescription' },
  { value: 'discharge_summary', label: 'Discharge Summary' },
  { value: 'consultation', label: 'Consultation Note' },
  { value: 'progress_note', label: 'Progress Note' },
  { value: 'operative_report', label: 'Operative Report' },
  { value: 'immunization', label: 'Immunization Record' },
  { value: 'referral', label: 'Referral' },
  { value: 'insurance', label: 'Insurance Document' },
  { value: 'consent_form', label: 'Consent Form' },
  { value: 'vital_signs', label: 'Vital Signs' },
  { value: 'allergy_record', label: 'Allergy Record' },
  { value: 'medication_list', label: 'Medication List' },
  { value: 'billing', label: 'Billing Statement' },
];

// Color palette for different field types/categories
const FIELD_COLORS = {
  // CBC Panel - Blues
  cbc: { border: '#3B82F6', bg: 'rgba(59, 130, 246, 0.15)', label: '#2563EB' },
  // Metabolic Panel - Greens
  metabolic: { border: '#10B981', bg: 'rgba(16, 185, 129, 0.15)', label: '#059669' },
  // Lipid Panel - Oranges
  lipid: { border: '#F59E0B', bg: 'rgba(245, 158, 11, 0.15)', label: '#D97706' },
  // Liver Panel - Purples
  liver: { border: '#8B5CF6', bg: 'rgba(139, 92, 246, 0.15)', label: '#7C3AED' },
  // Thyroid Panel - Teals
  thyroid: { border: '#14B8A6', bg: 'rgba(20, 184, 166, 0.15)', label: '#0D9488' },
  // Renal Panel - Indigos
  renal: { border: '#6366F1', bg: 'rgba(99, 102, 241, 0.15)', label: '#4F46E5' },
  // Cardiac Panel - Reds
  cardiac: { border: '#EF4444', bg: 'rgba(239, 68, 68, 0.15)', label: '#DC2626' },
  // Default - Gray
  default: { border: '#6B7280', bg: 'rgba(107, 114, 128, 0.15)', label: '#4B5563' },
};

// Map field names to categories
const getFieldCategory = (fieldName) => {
  const name = (fieldName || '').toLowerCase();

  // CBC Panel
  if (/^(wbc|rbc|hemoglobin|hgb|hematocrit|hct|mcv|mch|mchc|rdw|platelet|plt|mpv|neutro|lympho|mono|eosino|baso|bands|segs)/i.test(name)) {
    return 'cbc';
  }
  // Metabolic Panel
  if (/^(glucose|sodium|potassium|chloride|co2|bicarbonate|calcium|magnesium|phosph|anion)/i.test(name)) {
    return 'metabolic';
  }
  // Lipid Panel
  if (/^(cholesterol|hdl|ldl|vldl|triglyceride|lipid)/i.test(name)) {
    return 'lipid';
  }
  // Liver Panel
  if (/^(alt|ast|alp|alkaline|bilirubin|albumin|total.?protein|ggt|ld|ldh)/i.test(name)) {
    return 'liver';
  }
  // Thyroid Panel
  if (/^(tsh|t3|t4|thyroid|ft3|ft4|free.?t)/i.test(name)) {
    return 'thyroid';
  }
  // Renal Panel
  if (/^(bun|creatinine|egfr|gfr|urea|uric)/i.test(name)) {
    return 'renal';
  }
  // Cardiac Panel
  if (/^(troponin|bnp|ck|cpk|ck-mb|myoglobin|pro-bnp)/i.test(name)) {
    return 'cardiac';
  }

  return 'default';
};

const getFieldColor = (fieldName) => {
  const category = getFieldCategory(fieldName);
  return FIELD_COLORS[category] || FIELD_COLORS.default;
};

const DocumentDetail = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();

  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('structured');
  const [selectedField, setSelectedField] = useState(null);

  // PDF viewer state
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.0);

  // Reprocess & Delete state
  const [showReprocessMenu, setShowReprocessMenu] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState(false);
  const reprocessRef = useRef(null);

  // Close reprocess dropdown on outside click
  useEffect(() => {
    if (!showReprocessMenu) return;
    const handleClick = (e) => {
      if (reprocessRef.current && !reprocessRef.current.contains(e.target)) {
        setShowReprocessMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showReprocessMenu]);

  // Track actual rendered PDF page dimensions for overlay alignment
  const [pageDimensions, setPageDimensions] = useState({ width: 0, height: 0 });

  // Fetch job details
  const fetchJob = useCallback(async () => {
    try {
      const data = await getJobStatus(jobId);
      setJob(data);
    } catch (err) {
      console.error('Failed to fetch job:', err);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchJob();
  }, [fetchJob]);

  // Update page dimensions when scale changes
  useEffect(() => {
    // Small delay to let react-pdf re-render the page
    const timer = setTimeout(() => {
      const canvas = document.querySelector('.react-pdf__Page__canvas');
      if (canvas) {
        setPageDimensions({
          width: canvas.clientWidth,
          height: canvas.clientHeight
        });
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [scale, pageNumber]);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

  // Track actual rendered page dimensions for overlay alignment
  const onPageRenderSuccess = (page) => {
    // Get the actual rendered canvas dimensions
    const canvas = document.querySelector('.react-pdf__Page__canvas');
    if (canvas) {
      setPageDimensions({
        width: canvas.clientWidth,
        height: canvas.clientHeight
      });
    }
  };

  const handleFieldClick = (field) => {
    setSelectedField(field);
    // If field has bounding box, scroll to it
  };

  const handleBoxClick = (box) => {
    setSelectedField(box.field);
    // Navigate to the page containing this box
    if (box.page && box.page !== pageNumber) {
      setPageNumber(box.page);
    }
    setActiveTab('structured');
  };

  // Handle clicking a box from the list - navigate to page and highlight
  const handleBoxListClick = (box) => {
    setSelectedField(box.field);
    if (box.page && box.page !== pageNumber) {
      setPageNumber(box.page);
    }
    setActiveTab('boxes'); // Stay on boxes tab to show overlay
  };

  // Reprocess handler
  const handleReprocess = async (documentType = null) => {
    setShowReprocessMenu(false);
    setReprocessing(true);
    try {
      const result = await reprocessJob(jobId, documentType);
      // Navigate to dashboard to watch new processing
      navigate(`/dashboard?job=${result.job_id}`);
    } catch (err) {
      console.error('Failed to reprocess:', err);
      alert(err.response?.data?.detail || 'Failed to reprocess document.');
      setReprocessing(false);
    }
  };

  // Delete handler
  const handleDelete = async () => {
    if (!window.confirm('Delete this document? This cannot be undone.')) return;
    setDeletingDoc(true);
    try {
      await deleteJob(jobId);
      navigate('/documents');
    } catch (err) {
      console.error('Failed to delete:', err);
      alert('Failed to delete document.');
      setDeletingDoc(false);
    }
  };

  if (loading) {
    return (
      <div className="detail-loading">
        <Loader2 size={32} className="spinning" />
        <p>Loading document...</p>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="detail-error">
        <p>Document not found</p>
        <button className="btn btn-secondary" onClick={() => navigate('/documents')}>
          Back to Documents
        </button>
      </div>
    );
  }

  const result = job.result || {};
  const documentUrl = getDocumentUrl(job.file_path?.split('/').pop()?.split('.')[0] || job.job_id);

  return (
    <div className="document-detail">
      {/* Header */}
      <div className="detail-header">
        <div className="header-left">
          <button className="btn btn-ghost" onClick={() => navigate('/documents')}>
            <ArrowLeft size={20} />
            Back
          </button>
          <div className="header-info">
            <h1>{job.file_name}</h1>
            <div className="header-meta">
              <StatusBadge status={job.status} />
              {result.confidence && (
                <ConfidenceBadge score={result.confidence} />
              )}
              <span className="meta-item">{job.document_type}</span>
            </div>
          </div>
        </div>
        <div className="header-right">
          <button
            className="btn btn-secondary"
            onClick={() => navigate(`/flow/${jobId}`)}
          >
            <GitBranch size={18} />
            View Flow
          </button>
          <button className="btn btn-secondary">
            <Download size={18} />
            Export
          </button>

          {/* Reprocess Dropdown */}
          <div className="reprocess-wrapper" ref={reprocessRef}>
            <button
              className="btn btn-secondary"
              onClick={() => setShowReprocessMenu(!showReprocessMenu)}
              disabled={reprocessing}
            >
              {reprocessing
                ? <Loader2 size={18} className="spinning" />
                : <RotateCcw size={18} />
              }
              Reprocess
              <ChevronDown size={14} />
            </button>
            {showReprocessMenu && (
              <div className="reprocess-dropdown">
                <button
                  className="dropdown-item"
                  onClick={() => handleReprocess(null)}
                >
                  Reprocess (auto-classify)
                </button>
                <div className="dropdown-divider" />
                <div className="dropdown-header">Reclassify as:</div>
                {RECLASSIFY_TYPES
                  .filter(t => t.value !== (job.document_type || '').toLowerCase())
                  .map(type => (
                    <button
                      key={type.value}
                      className="dropdown-item"
                      onClick={() => handleReprocess(type.value)}
                    >
                      {type.label}
                    </button>
                  ))
                }
              </div>
            )}
          </div>

          {/* Delete */}
          <button
            className="btn btn-danger-ghost"
            onClick={handleDelete}
            disabled={deletingDoc}
            title="Delete document"
          >
            {deletingDoc
              ? <Loader2 size={18} className="spinning" />
              : <Trash2 size={18} />
            }
          </button>
        </div>
      </div>

      {/* Split View */}
      <div className="split-view">
        {/* Left: PDF Viewer */}
        <div className="pdf-panel">
          <div className="pdf-toolbar">
            <div className="toolbar-left">
              <span className="page-info">
                Page {pageNumber} of {numPages || '?'}
              </span>
            </div>
            <div className="toolbar-center">
              <button
                className="btn btn-ghost"
                onClick={() => setPageNumber(Math.max(1, pageNumber - 1))}
                disabled={pageNumber <= 1}
              >
                <ChevronLeft size={20} />
              </button>
              <button
                className="btn btn-ghost"
                onClick={() => setPageNumber(Math.min(numPages || 1, pageNumber + 1))}
                disabled={pageNumber >= numPages}
              >
                <ChevronRight size={20} />
              </button>
            </div>
            <div className="toolbar-right">
              <button
                className="btn btn-ghost"
                onClick={() => setScale(Math.max(0.5, scale - 0.25))}
              >
                <ZoomOut size={20} />
              </button>
              <span className="zoom-level">{Math.round(scale * 100)}%</span>
              <button
                className="btn btn-ghost"
                onClick={() => setScale(Math.min(2, scale + 0.25))}
              >
                <ZoomIn size={20} />
              </button>
            </div>
          </div>

          <div className="pdf-container">
            <div className="pdf-page-wrapper" style={{ position: 'relative', display: 'inline-block' }}>
              <Document
                file={documentUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                loading={
                  <div className="pdf-loading">
                    <Loader2 size={24} className="spinning" />
                  </div>
                }
                error={
                  <div className="pdf-error">
                    <p>Failed to load document</p>
                  </div>
                }
              >
                <Page
                  pageNumber={pageNumber}
                  scale={scale}
                  renderTextLayer={false}
                  renderAnnotationLayer={false}
                  onRenderSuccess={onPageRenderSuccess}
                />
              </Document>

              {/* Bounding Box Overlay - positioned to match PDF page dimensions */}
              {activeTab === 'boxes' && result.bounding_boxes && pageDimensions.width > 0 && (
                <div
                  className="bbox-overlay"
                  style={{
                    width: pageDimensions.width,
                    height: pageDimensions.height,
                  }}
                >
                  {result.bounding_boxes
                    .filter(box => !box.page || box.page === pageNumber)
                    .map((box, index) => {
                      const colors = getFieldColor(box.field);
                      const isSelected = selectedField === box.field;
                      return (
                        <div
                          key={index}
                          className={`bbox ${isSelected ? 'selected' : ''}`}
                          style={{
                            left: `${box.x}%`,
                            top: `${box.y}%`,
                            width: `${box.width}%`,
                            height: `${box.height}%`,
                            borderColor: isSelected ? '#1F2937' : colors.border,
                            backgroundColor: isSelected ? 'rgba(31, 41, 55, 0.25)' : colors.bg,
                          }}
                          onClick={() => handleBoxClick(box)}
                          title={`${box.field}: ${box.value}`}
                        >
                          <span
                            className="bbox-label"
                            style={{ backgroundColor: isSelected ? '#1F2937' : colors.label }}
                          >
                            {box.field}
                          </span>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Data Panel */}
        <div className="data-panel">
          <div className="tabs">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="tab-content">
            {activeTab === 'structured' && (
              <StructuredDataView
                result={result}
                documentType={job.document_type}
                selectedField={selectedField}
                onFieldClick={handleFieldClick}
                boundingBoxes={result.bounding_boxes || []}
                onNavigateToBox={(box) => {
                  setSelectedField(box.field);
                  if (box.page) setPageNumber(box.page);
                  setActiveTab('boxes');
                }}
              />
            )}
            {activeTab === 'formatted' && (
              <FormattedView
                result={result}
                documentType={job.document_type}
              />
            )}
            {activeTab === 'json' && (
              <JsonView data={result.universal_extraction || result} />
            )}
            {activeTab === 'fhir' && (
              <FhirView data={result.fhir_bundle} />
            )}
            {activeTab === 'boxes' && (
              <BoundingBoxesView
                boxes={result.bounding_boxes || []}
                selectedField={selectedField}
                onBoxClick={handleBoxListClick}
                currentPage={pageNumber}
              />
            )}
            {activeTab === 'chat' && (
              <ChatView jobId={jobId} documentType={job.document_type} />
            )}
            {activeTab === 'workflow' && (
              <WorkflowLogView steps={job.workflow_steps || []} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Tab Content Components
// ============================================================================

const StructuredDataView = ({ result, documentType, selectedField, onFieldClick, boundingBoxes, onNavigateToBox }) => {
  // Debug: log the incoming result
  console.log('StructuredDataView - documentType:', documentType);
  console.log('StructuredDataView - result.extracted_values:', result.extracted_values?.length, result.extracted_values);
  console.log('StructuredDataView - result.universal_extraction.test_results:', result.universal_extraction?.test_results?.length);

  // Find bounding box for a field name
  const findBoxForField = (fieldName) => {
    return boundingBoxes?.find(box => box.field === fieldName);
  };

  const handleRowClick = (item) => {
    onFieldClick(item.field_name);
    const box = findBoxForField(item.field_name);
    if (box && onNavigateToBox) {
      onNavigateToBox(box);
    }
  };

  // Get raw_fields for tabular display (v2 format)
  const rawFields = result.raw_fields || {};
  const hasRawFields = Object.keys(rawFields).length > 0;

  // Get extracted_values (array format)
  const extractedValues = Array.isArray(result.extracted_values) ? result.extracted_values : [];

  const renderRawFieldsTable = () => {
    if (!hasRawFields) return null;

    return (
      <div className="structured-section">
        <h3>Extracted Fields ({Object.keys(rawFields).length})</h3>
        <div className="results-table">
          <table className="table fields-table">
            <thead>
              <tr>
                <th>Field</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(rawFields).map(([key, value]) => (
                <tr key={key}>
                  <td className="field-name">{key.replace(/_/g, ' ')}</td>
                  <td className="field-value">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderLabResults = () => {
    const values = extractedValues.length > 0 ? extractedValues : (result.extracted_values || []);

    if (values.length === 0 && !hasRawFields) {
      return renderRawFieldsTable();
    }

    return (
      <div className="structured-sections">
        {values.length > 0 && (
          <div className="structured-section">
            <h3>Lab Results</h3>
            <div className="results-table">
              <table className="table">
                <thead>
                  <tr>
                    <th>Test</th>
                    <th>Value</th>
                    <th>Unit</th>
                    <th>Reference</th>
                    <th>Flag</th>
                    <th>Confidence</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {values.map((item, index) => {
                    const hasBox = !!findBoxForField(item.field_name);
                    return (
                      <tr
                        key={index}
                        className={`clickable ${selectedField === item.field_name ? 'selected' : ''} ${hasBox ? 'has-bbox' : ''}`}
                        onClick={() => handleRowClick(item)}
                      >
                        <td>{item.field_name || item.name || '-'}</td>
                        <td className="value-cell">{item.value}</td>
                        <td>{item.unit || '-'}</td>
                        <td>
                          {item.reference_min && item.reference_max
                            ? `${item.reference_min} - ${item.reference_max}`
                            : '-'}
                        </td>
                        <td>
                          {item.abnormal_flag && (
                            <span className={`flag flag-${String(item.abnormal_flag).toLowerCase()}`}>
                              {item.abnormal_flag}
                            </span>
                          )}
                        </td>
                        <td>
                          <ConfidenceBadge score={item.confidence} showLabel={false} size="sm" />
                        </td>
                        <td className="bbox-indicator">
                          {hasBox && <span className="bbox-dot" title="Click to view on PDF">📍</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {renderRawFieldsTable()}
      </div>
    );
  };

  const renderRadiologyResults = () => {
    const sections = result.sections || {};

    return (
      <div className="structured-sections">
        {result.clinical_summary && (
          <div className="structured-section">
            <h3>Impression</h3>
            <p className="section-text">{result.clinical_summary}</p>
          </div>
        )}
        {sections.findings && (
          <div className="structured-section">
            <h3>Findings</h3>
            {Array.isArray(sections.findings) ? (
              <ul className="findings-list">
                {sections.findings.map((finding, index) => (
                  <li key={index}>{typeof finding === 'object' ? finding.description || JSON.stringify(finding) : finding}</li>
                ))}
              </ul>
            ) : (
              <p className="section-text">{sections.findings}</p>
            )}
          </div>
        )}
        {result.critical_findings?.length > 0 && (
          <div className="structured-section critical">
            <h3>Critical Findings</h3>
            <ul className="critical-list">
              {result.critical_findings.map((finding, index) => (
                <li key={index}>{finding}</li>
              ))}
            </ul>
          </div>
        )}
        {renderRawFieldsTable()}
      </div>
    );
  };

  const renderPrescriptionResults = () => {
    const sections = result.sections || {};
    const medications = sections.medications || [];
    const universal = result.universal_extraction || {};
    const meds = universal.medications || medications;

    return (
      <div className="structured-sections">
        {meds.length > 0 && (
          <div className="structured-section">
            <h3>Medications</h3>
            {meds.map((med, index) => {
              const validationStatus = med.validation_status || '';
              const statusConfig = {
                'verified': { label: 'Verified', className: 'status-verified', icon: '✓' },
                'ocr_corrected': { label: 'OCR Corrected', className: 'status-ocr-corrected', icon: '~' },
                'medgemma_verified': { label: 'AI Verified', className: 'status-ai-verified', icon: '◆' },
                'strength_mismatch': { label: 'Strength Mismatch', className: 'status-strength-mismatch', icon: '⚠' },
                'unverified': { label: 'Unverified', className: 'status-unverified', icon: '!' },
              };
              const status = statusConfig[validationStatus];

              const rawName = med.medication_name || med.name || 'Unknown';
              const displayName = med.rxnorm_name || rawName;
              const wasCorrected = med.rxnorm_name && med.rxnorm_name.toLowerCase() !== rawName.toLowerCase()
                && !rawName.toLowerCase().startsWith(med.rxnorm_name.toLowerCase());

              return (
                <div key={index} className={`medication-card ${validationStatus === 'unverified' || validationStatus === 'strength_mismatch' ? 'med-unverified' : ''}`}>
                  <div className="med-header">
                    <div className="med-name">{displayName}</div>
                    {status && (
                      <span className={`med-validation-badge ${status.className}`}>
                        {status.icon} {status.label}
                      </span>
                    )}
                  </div>
                  <div className="med-details">
                    <span>{med.strength || med.dosage || ''}</span>
                    <span>{med.route || ''}</span>
                    <span>{med.frequency || ''}</span>
                  </div>
                  {wasCorrected && (
                    <div className="med-correction">
                      Originally read as: <strong>{rawName}</strong>
                    </div>
                  )}
                  {med.loinc_name && !(rawName).toLowerCase().startsWith(med.loinc_name.toLowerCase()) && (
                    <div className="med-correction">
                      Matched to: <strong>{med.loinc_name}</strong> (LOINC)
                    </div>
                  )}
                  {validationStatus === 'unverified' && (
                    <div className="med-warning">
                      <AlertTriangle size={14} /> Not found in drug database — manual review required
                    </div>
                  )}
                  {validationStatus === 'strength_mismatch' && (
                    <div className="med-warning">
                      <AlertTriangle size={14} /> Strength does not match known dosages for this drug — possible misread
                    </div>
                  )}
                  <div className="med-meta">
                    {med.quantity && <span>Qty: {med.quantity}</span>}
                    {med.refills !== undefined && <span> | Refills: {med.refills}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {sections.drug_interactions?.length > 0 && (
          <div className="structured-section warning">
            <h3>Drug Interactions</h3>
            <ul>
              {sections.drug_interactions.map((interaction, index) => (
                <li key={index}>{interaction}</li>
              ))}
            </ul>
          </div>
        )}
        {renderRawFieldsTable()}
      </div>
    );
  };

  const renderGenericResults = () => {
    // Debug: log what we're rendering
    console.log('renderGenericResults - extractedValues:', extractedValues.length, extractedValues);
    console.log('renderGenericResults - rawFields:', Object.keys(rawFields).length, rawFields);

    return (
      <div className="structured-sections">
        {/* Show extracted values FIRST (more important than raw fields) */}
        {extractedValues.length > 0 && (
          <div className="structured-section">
            <h3>Extracted Values ({extractedValues.length})</h3>
            <div className="results-table">
              <table className="table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Value</th>
                    <th>Unit</th>
                    <th>Reference</th>
                    <th>Flag</th>
                    <th>Category</th>
                  </tr>
                </thead>
                <tbody>
                  {extractedValues.map((item, index) => (
                    <tr key={index}>
                      <td>{item.field_name || item.name || '-'}</td>
                      <td className="value-cell">{item.value || '-'}</td>
                      <td>{item.unit || '-'}</td>
                      <td>
                        {item.reference_min && item.reference_max
                          ? `${item.reference_min} - ${item.reference_max}`
                          : '-'}
                      </td>
                      <td>
                        {item.abnormal_flag && (
                          <span className={`flag flag-${String(item.abnormal_flag).toLowerCase()}`}>
                            {item.abnormal_flag}
                          </span>
                        )}
                      </td>
                      <td>{item.category || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Show raw fields AFTER extracted values */}
        {renderRawFieldsTable()}

        {/* Show sections */}
        {result.sections && Object.entries(result.sections)
          .filter(([name]) => name !== 'extracted_fields')
          .map(([name, content]) => (
            <div key={name} className="structured-section">
              <h3>{name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
              {typeof content === 'object' && !Array.isArray(content) ? (
                <div className="key-value-list">
                  {Object.entries(content).map(([k, v]) => (
                    <div key={k} className="kv-row">
                      <span className="kv-key">{k.replace(/_/g, ' ')}:</span>
                      <span className="kv-value">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
                    </div>
                  ))}
                </div>
              ) : Array.isArray(content) ? (
                <ul className="section-list">
                  {content.map((item, i) => (
                    <li key={i}>{typeof item === 'object' ? JSON.stringify(item) : String(item)}</li>
                  ))}
                </ul>
              ) : (
                <p className="section-text">{String(content)}</p>
              )}
            </div>
          ))}
      </div>
    );
  };

  // Normalize document type for comparison (case-insensitive)
  const normalizedType = (documentType || '').toLowerCase();

  // Determine if this looks like a lab document based on content
  // (has test results with values, units, reference ranges)
  const hasLabLikeContent = extractedValues.length > 0 &&
    extractedValues.some(v => v.unit || v.reference_min || v.reference_max);

  return (
    <div className="structured-data">
      {result.requires_review && (
        <div className="review-banner">
          <span>This document requires human review</span>
        </div>
      )}

      {/* Show lab results for 'lab' type OR any document with lab-like extracted values */}
      {(normalizedType === 'lab' || (hasLabLikeContent && !['radiology', 'prescription'].includes(normalizedType))) && renderLabResults()}
      {normalizedType === 'radiology' && renderRadiologyResults()}
      {normalizedType === 'prescription' && renderPrescriptionResults()}

      {/* Generic fallback for other document types without lab-like content */}
      {!['lab', 'radiology', 'prescription'].includes(normalizedType) && !hasLabLikeContent && renderGenericResults()}
    </div>
  );
};

// Formatted View - Template-based rendering based on document type
const FormattedView = ({ result, documentType }) => {
  const universal = result.universal_extraction || {};
  const classification = result.classification || {};
  const rawFields = result.raw_fields || {};

  // For FHIR - we use enriched data if available, otherwise raw extraction
  // This answers the user's question: FHIR should read from formatted/enriched data first

  const renderInsuranceDocument = () => {
    return (
      <div className="formatted-document insurance">
        <div className="doc-header">
          <h2>Insurance Claim Document</h2>
          <span className="doc-type-badge">{classification.type || documentType}</span>
        </div>

        <div className="doc-section">
          <h3>Claim Information</h3>
          <div className="info-grid">
            {rawFields.claim_type && (
              <div className="info-item">
                <label>Claim Type</label>
                <span className="value highlight">{rawFields.claim_type}</span>
              </div>
            )}
            {rawFields.benefit_type && (
              <div className="info-item">
                <label>Benefit Type</label>
                <span className="value">{rawFields.benefit_type}</span>
              </div>
            )}
            {rawFields.submission_date && (
              <div className="info-item">
                <label>Submission Date</label>
                <span className="value">{rawFields.submission_date}</span>
              </div>
            )}
            {rawFields.invoice_number && (
              <div className="info-item">
                <label>Invoice Number</label>
                <span className="value">{rawFields.invoice_number}</span>
              </div>
            )}
          </div>
        </div>

        {(rawFields.participant || rawFields.member_id) && (
          <div className="doc-section">
            <h3>Participant Information</h3>
            <div className="info-grid">
              {rawFields.participant && (
                <div className="info-item">
                  <label>Participant</label>
                  <span className="value">{rawFields.participant}</span>
                </div>
              )}
              {rawFields.member_id && (
                <div className="info-item">
                  <label>Member ID</label>
                  <span className="value">{rawFields.member_id}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {rawFields.total_amount && (
          <div className="doc-section">
            <h3>Financial Summary</h3>
            <div className="info-grid">
              <div className="info-item large">
                <label>Total Amount</label>
                <span className="value amount">{rawFields.total_amount}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderLabDocument = () => {
    const testResults = universal.test_results || [];

    return (
      <div className="formatted-document lab">
        <div className="doc-header">
          <h2>Laboratory Report</h2>
          <span className="doc-type-badge">Lab Results</span>
        </div>

        {universal.patient_info && (
          <div className="doc-section">
            <h3>Patient Information</h3>
            <div className="info-grid">
              {Object.entries(universal.patient_info).map(([k, v]) => (
                <div key={k} className="info-item">
                  <label>{k.replace(/_/g, ' ')}</label>
                  <span className="value">{String(v)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {testResults.length > 0 && (
          <div className="doc-section">
            <h3>Test Results</h3>
            <table className="formatted-table">
              <thead>
                <tr>
                  <th>Test</th>
                  <th>Result</th>
                  <th>Unit</th>
                  <th>Reference Range</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {testResults.map((test, i) => (
                  <tr key={i} className={test.flag ? 'abnormal' : ''}>
                    <td>{test.name}</td>
                    <td className="result-value">{test.value}</td>
                    <td>{test.unit || '-'}</td>
                    <td>{test.reference_min && test.reference_max ? `${test.reference_min} - ${test.reference_max}` : '-'}</td>
                    <td>{test.flag ? <span className="flag-badge">{test.flag}</span> : 'Normal'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  };

  const renderGenericDocument = () => {
    return (
      <div className="formatted-document generic">
        <div className="doc-header">
          <h2>Document Summary</h2>
          <span className="doc-type-badge">{classification.type || documentType || 'Unknown'}</span>
        </div>

        {Object.keys(rawFields).length > 0 && (
          <div className="doc-section">
            <h3>Extracted Information</h3>
            <div className="info-grid wide">
              {Object.entries(rawFields).map(([key, value]) => (
                <div key={key} className="info-item">
                  <label>{key.replace(/_/g, ' ')}</label>
                  <span className="value">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.clinical_summary && (
          <div className="doc-section">
            <h3>Summary</h3>
            <p className="summary-text">{result.clinical_summary}</p>
          </div>
        )}
      </div>
    );
  };

  // Render based on document type
  const docType = (documentType || '').toLowerCase();

  if (docType.includes('insurance') || rawFields.claim_type || rawFields.benefit_type) {
    return renderInsuranceDocument();
  }

  if (docType === 'lab') {
    return renderLabDocument();
  }

  return renderGenericDocument();
};

const JsonView = ({ data }) => (
  <div className="json-view">
    <pre className="json-content">
      {JSON.stringify(data, null, 2)}
    </pre>
  </div>
);

const FhirView = ({ data }) => (
  <div className="fhir-view">
    {data ? (
      <pre className="json-content">
        {JSON.stringify(data, null, 2)}
      </pre>
    ) : (
      <div className="empty-state">
        <p>FHIR bundle not available</p>
      </div>
    )}
  </div>
);

const BoundingBoxesView = ({ boxes, selectedField, onBoxClick, currentPage }) => {
  // Group boxes by category for legend
  const categories = [...new Set(boxes.map(b => getFieldCategory(b.field)))];

  return (
    <div className="boxes-view">
      <p className="boxes-hint">Click a box to navigate to its location on the document</p>

      {/* Color Legend */}
      {categories.length > 1 && (
        <div className="boxes-legend">
          {categories.map(cat => {
            const colors = FIELD_COLORS[cat];
            return (
              <span key={cat} className="legend-item">
                <span
                  className="legend-color"
                  style={{ backgroundColor: colors.border }}
                />
                <span className="legend-label">{cat.toUpperCase()}</span>
              </span>
            );
          })}
        </div>
      )}

      <div className="boxes-stats">
        <span>{boxes.length} bounding boxes found</span>
        {currentPage && (
          <span className="current-page-indicator">
            Page {currentPage}: {boxes.filter(b => !b.page || b.page === currentPage).length} boxes
          </span>
        )}
      </div>
      <div className="boxes-list">
        {boxes.length === 0 ? (
          <div className="empty-state">
            <p>No bounding boxes available</p>
            <p className="empty-hint">Bounding boxes are extracted from table cells and OCR results</p>
          </div>
        ) : (
          boxes.map((box, index) => {
            const colors = getFieldColor(box.field);
            const isSelected = selectedField === box.field;
            return (
              <div
                key={index}
                className={`box-item ${isSelected ? 'selected' : ''} ${box.page === currentPage ? 'on-current-page' : ''}`}
                onClick={() => onBoxClick(box)}
                style={{
                  borderLeftColor: isSelected ? '#1F2937' : colors.border,
                }}
              >
                <div className="box-main">
                  <span
                    className="box-color-indicator"
                    style={{ backgroundColor: colors.border }}
                  />
                  <span className="box-field">{box.field}</span>
                  <span className="box-value">{box.value}</span>
                </div>
                <div className="box-meta">
                  {box.page && <span className="box-page">Page {box.page}</span>}
                  <ConfidenceBadge score={box.confidence || 0.9} showLabel={false} size="sm" />
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

const WorkflowLogView = ({ steps }) => {
  // Format stage name for display
  const formatStageName = (name) => {
    const nameMap = {
      'text_extraction': 'Text Extraction',
      'raw_field_extraction': 'Field Pre-Extraction',
      'similar_doc_lookup': 'Similar Document Lookup',
      'parallel_extraction': 'AI Extraction + Classification',
      'vector_store': 'Vector Store Update',
      'lab_enrichment': 'Lab Enrichment',
      'prescription_enrichment': 'Prescription Enrichment',
      'radiology_enrichment': 'Radiology Enrichment',
      'pathology_enrichment': 'Pathology Enrichment',
      'database_validation': 'Database Validation',
      // Cloud path stages
      'document_intake': 'Document Intake',
      'cloud_extraction': 'GPT-4o Vision API Call',
    };
    return nameMap[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Get icon for stage
  const getStageIcon = (name) => {
    const iconMap = {
      'text_extraction': '📄',
      'raw_field_extraction': '🔍',
      'similar_doc_lookup': '🔗',
      'parallel_extraction': '🤖',
      'vector_store': '💾',
      'lab_enrichment': '🧪',
      'prescription_enrichment': '💊',
      'radiology_enrichment': '📷',
      'pathology_enrichment': '🔬',
      'database_validation': '🛡️',
      // Cloud path stages
      'document_intake': '📋',
      'cloud_extraction': '☁️',
    };
    return iconMap[name] || '⚙️';
  };

  // Render detail items
  const renderDetails = (details) => {
    if (!details || Object.keys(details).length === 0) return null;

    // Priority order for displaying details
    const priorityKeys = [
      'description',
      'llm_model', 'vlm_model', 'embedding_model', 'deployment',
      'document_type', 'classification_confidence', 'reasoning',
      'submission_type', 'claim_type', 'line_of_benefits', 'benefit_type',
      'method', 'target_api',
      'chars_extracted', 'pages', 'pages_processed', 'file_size_kb',
      'fields_extracted', 'extraction_confidence',
      'test_results_count', 'medications_count', 'findings_count',
      'procedures_count', 'dates_count', 'providers_count', 'organizations_count',
      'patient_extracted', 'raw_fields_count',
      'conversion_time', 'api_call_time',
      'similar_docs_found', 'strategy_used',
      'max_tokens', 'temperature',
      // Database validation keys
      'database', 'total_items_checked', 'verified_count', 'unverified_count', 'strength_mismatch_count',
      'rxnorm_codes_found', 'loinc_codes_found', 'ocr_corrections', 'interactions_detected',
      'warning'
    ];

    // Format value for display
    const formatValue = (key, value) => {
      if (value === null || value === undefined) return null;
      if (typeof value === 'object' && !Array.isArray(value)) return null;  // Skip nested objects
      if (typeof value === 'boolean') return value ? 'Yes' : 'No';
      if (key.includes('confidence')) return `${Math.round(value * 100)}%`;
      if (key === 'chars_extracted') return `${value.toLocaleString()} chars`;
      if (key === 'file_size_kb') return `${value} KB`;
      if (key.endsWith('_time') && typeof value === 'number') return `${value.toFixed(1)}s`;
      if (key === 'temperature') return String(value);
      if (Array.isArray(value)) return value.slice(0, 5).join(', ') + (value.length > 5 ? '...' : '');
      return String(value);
    };

    // Format key for display
    const formatKey = (key) => {
      const keyMap = {
        'llm_model': 'LLM Model',
        'vlm_model': 'VLM Model',
        'embedding_model': 'Embedding Model',
        'document_type': 'Document Type',
        'classification_confidence': 'Classification Confidence',
        'chars_extracted': 'Characters',
        'pages': 'Pages',
        'source_type': 'Source',
        'method': 'Method',
        'fields_extracted': 'Fields Found',
        'field_names': 'Field Names',
        'similar_docs_found': 'Similar Docs',
        'strategy_used': 'Strategy',
        'test_results_count': 'Test Results',
        'medications_count': 'Medications',
        'findings_count': 'Findings',
        'patient_extracted': 'Patient Info',
        'raw_fields_count': 'Raw Fields',
        'file_type': 'File Type',
        'file_name': 'File Name',
        'file_size_kb': 'File Size',
        'ocr_used': 'OCR Used',
        'has_hints': 'Using Hints',
        'parallel': 'Parallel Processing',
        'description': 'Description',
        // Cloud path detail keys
        'deployment': 'Model Deployment',
        'target_api': 'Target API',
        'pages_processed': 'Pages Processed',
        'conversion_time': 'PDF Conversion',
        'api_call_time': 'API Call Time',
        'max_tokens': 'Max Tokens',
        'temperature': 'Temperature',
        'extraction_confidence': 'Extraction Confidence',
        'procedures_count': 'Procedures',
        'dates_count': 'Dates',
        'providers_count': 'Providers',
        'organizations_count': 'Organizations',
        'reasoning': 'Reasoning',
        'submission_type': 'Submission Type',
        'claim_type': 'Claim Type',
        'line_of_benefits': 'Line of Benefits',
        'benefit_type': 'Benefit Type',
        // Database validation keys
        'database': 'Database',
        'total_items_checked': 'Items Checked',
        'verified_count': 'Verified',
        'unverified_count': 'Unverified',
        'strength_mismatch_count': 'Strength Mismatch',
        'rxnorm_codes_found': 'RxNorm Codes',
        'loinc_codes_found': 'LOINC Codes',
        'ocr_corrections': 'OCR Corrections',
        'interactions_detected': 'Drug Interactions',
        'warning': 'Warning',
      };
      return keyMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    };

    // Sort and filter details
    const sortedKeys = Object.keys(details).sort((a, b) => {
      const aIdx = priorityKeys.indexOf(a);
      const bIdx = priorityKeys.indexOf(b);
      if (aIdx === -1 && bIdx === -1) return 0;
      if (aIdx === -1) return 1;
      if (bIdx === -1) return -1;
      return aIdx - bIdx;
    });

    return (
      <div className="log-details">
        {sortedKeys.map(key => {
          const value = formatValue(key, details[key]);
          if (value === null) return null;

          // Highlight model names
          const isModel = key.includes('model');

          return (
            <div key={key} className={`detail-item ${isModel ? 'model' : ''}`}>
              <span className="detail-key">{formatKey(key)}:</span>
              <span className={`detail-value ${isModel ? 'model-name' : ''}`}>{value}</span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="workflow-log">
      <div className="workflow-header">
        <h3>Document Processing Pipeline</h3>
        <p className="workflow-subtitle">Detailed view of the extraction lifecycle</p>
      </div>

      {steps.length === 0 ? (
        <div className="empty-state">
          <p>No workflow steps recorded</p>
        </div>
      ) : (
        <div className="log-entries">
          {steps.map((step, index) => {
            const duration = step.duration_seconds ||
              (step.completed_at && step.started_at
                ? (new Date(step.completed_at) - new Date(step.started_at)) / 1000
                : null);

            return (
              <div key={index} className={`log-entry ${step.status}`}>
                <div className="log-connector">
                  <div className="connector-line top" />
                  <div className={`connector-dot ${step.status}`}>
                    <span className="stage-icon">{getStageIcon(step.name)}</span>
                  </div>
                  <div className="connector-line bottom" />
                </div>

                <div className="log-card">
                  <div className="log-header">
                    <div className="log-title">
                      <span className="log-name">{formatStageName(step.name)}</span>
                      <span className={`log-status-badge ${step.status}`}>{step.status}</span>
                    </div>
                    <div className="log-meta">
                      {step.started_at && (
                        <span className="log-time">
                          {new Date(step.started_at).toLocaleTimeString()}
                        </span>
                      )}
                      {duration !== null && (
                        <span className="log-duration">
                          {duration < 1 ? `${Math.round(duration * 1000)}ms` : `${duration.toFixed(1)}s`}
                        </span>
                      )}
                    </div>
                  </div>

                  {step.details && renderDetails(step.details)}

                  {step.error && (
                    <div className="log-error">
                      <span className="error-label">Error:</span>
                      <span className="error-message">{step.error}</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Chat View - Ask questions about extracted document data
// ============================================================================

const ChatView = ({ jobId, documentType }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load suggestions on mount
  useEffect(() => {
    const loadSuggestions = async () => {
      try {
        const data = await getChatSuggestions(jobId);
        setSuggestions(data.suggestions || []);
      } catch {
        setSuggestions([
          'Summarize this document for me',
          'Are there any abnormal findings?',
          'What information was extracted?',
        ]);
      }
    };
    loadSuggestions();
  }, [jobId]);

  const handleSend = async (text) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;

    setMessages((prev) => [...prev, { role: 'user', content: msg }]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const data = await sendChatMessage(jobId, msg);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.reply, model: data.model_used },
      ]);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to send message. Make sure the backend is running.');
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleSend();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = async () => {
    try {
      await clearChatHistory(jobId);
      setMessages([]);
      setError(null);
    } catch (err) {
      console.error('Failed to clear chat history:', err);
    }
  };

  return (
    <div className="chat-view">
      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <div className="chat-empty">
            <div className="chat-empty-icon">💬</div>
            <h3>Ask about this document</h3>
            <p>Ask questions about the extracted data, test results, medications, or findings.</p>

            {suggestions.length > 0 && (
              <div className="chat-suggestions">
                {suggestions.map((s, i) => (
                  <button
                    key={i}
                    className="chat-suggestion-btn"
                    onClick={() => handleSend(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-message chat-message-${msg.role}`}>
            <div className="chat-message-avatar">
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="chat-message-content">
              <div className="chat-message-text">{msg.content}</div>
              {msg.model && (
                <span className="chat-message-model">{msg.model}</span>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message chat-message-assistant">
            <div className="chat-message-avatar">🤖</div>
            <div className="chat-message-content">
              <div className="chat-typing">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}

        {error && <div className="chat-error">{error}</div>}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        {messages.length > 0 && (
          <button className="chat-clear-btn" onClick={handleClear}>
            Clear history
          </button>
        )}
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about this document..."
            disabled={loading}
          />
          <button
            type="submit"
            className="chat-send-btn"
            disabled={!input.trim() || loading}
          >
            {loading ? <Loader2 size={18} className="spinning" /> : <Send size={18} />}
          </button>
        </form>
        <p className="chat-disclaimer">
          AI-assisted information only. Consult a healthcare provider for medical decisions.
        </p>
      </div>
    </div>
  );
};

export default DocumentDetail;
