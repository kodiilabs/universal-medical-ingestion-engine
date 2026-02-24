import React, { useState } from 'react';
import './ExtractedData.css';

function ExtractedData({ data, isProcessing, workflowSteps }) {
  const [activeTab, setActiveTab] = useState('structured');

  const formatStepName = (name) => {
    const nameMap = {
      'text_extraction': 'Text Extraction',
      'raw_field_extraction': 'Field Pre-Extraction',
      'similar_doc_lookup': 'Similar Document Lookup',
      'parallel_extraction': 'AI Extraction + Classification',
      'vector_store': 'Vector Store Update',
      'document_intake': 'Document Intake',
      'cloud_extraction': 'GPT-4o Vision API Call',
    };
    return nameMap[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  if (isProcessing) {
    return (
      <div className="extracted-data">
        <div className="processing-status">
          <div className="spinner"></div>
          <h3>Processing Document...</h3>
          <div className="workflow-progress">
            {workflowSteps.map((step, index) => (
              <div key={step.id} className={`step ${step.status}`}>
                <div className="step-indicator">
                  {step.status === 'completed' ? '✓' : step.status === 'running' ? '...' : '○'}
                </div>
                <span>{formatStepName(step.name)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="extracted-data empty">
        <div className="empty-state">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            <path d="M9 14l2 2 4-4" />
          </svg>
          <p>Process a document to see extracted data</p>
        </div>
      </div>
    );
  }

  const renderConfidenceBadge = (confidence) => {
    const percent = Math.round((confidence || 0) * 100);
    const color = percent >= 85 ? '#4ade80' : percent >= 70 ? '#fbbf24' : '#f87171';
    return (
      <span className="confidence-badge" style={{ backgroundColor: color }}>
        {percent}% confidence
      </span>
    );
  };

  const renderValue = (value, depth = 0) => {
    if (value === null || value === undefined) return <span className="null">null</span>;
    if (typeof value === 'boolean') return <span className="boolean">{value.toString()}</span>;
    if (typeof value === 'number') return <span className="number">{value}</span>;
    if (typeof value === 'string') return <span className="string">"{value}"</span>;

    if (Array.isArray(value)) {
      if (value.length === 0) return <span className="array">[]</span>;
      return (
        <div className="array" style={{ marginLeft: depth * 16 }}>
          [
          {value.map((item, i) => (
            <div key={i} className="array-item">
              {renderValue(item, depth + 1)}
              {i < value.length - 1 && ','}
            </div>
          ))}
          ]
        </div>
      );
    }

    if (typeof value === 'object') {
      const entries = Object.entries(value);
      if (entries.length === 0) return <span className="object">{'{}'}</span>;
      return (
        <div className="object" style={{ marginLeft: depth * 16 }}>
          {'{'}
          {entries.map(([key, val], i) => (
            <div key={key} className="object-entry">
              <span className="key">"{key}"</span>: {renderValue(val, depth + 1)}
              {i < entries.length - 1 && ','}
            </div>
          ))}
          {'}'}
        </div>
      );
    }

    return String(value);
  };

  const renderStructuredView = () => {
    // Get raw_fields for tabular display (v2 format)
    const rawFields = data.raw_fields || {};
    const hasRawFields = Object.keys(rawFields).length > 0;

    // Get extracted_values (array format)
    const extractedValues = Array.isArray(data.extracted_values) ? data.extracted_values : [];

    // Get classification info
    const classification = data.classification || {};
    const docType = classification.type || data.document_type || 'unknown';

    return (
      <div className="structured-view">
        {/* Classification & Summary */}
        <div className="data-section">
          <div className="section-header">
            <h3>Document Summary</h3>
            {renderConfidenceBadge(data.confidence)}
          </div>
          <div className="section-content">
            <div className="summary-grid">
              <div className="field">
                <label>Document Type</label>
                <span className="doc-type-badge">{docType}</span>
              </div>
              {classification.confidence && (
                <div className="field">
                  <label>Classification Confidence</label>
                  <span>{Math.round(classification.confidence * 100)}%</span>
                </div>
              )}
            </div>
            {data.clinical_summary && (
              <div className="field">
                <label>Clinical Summary</label>
                <p>{data.clinical_summary}</p>
              </div>
            )}
            {data.requires_review && (
              <div className="review-badge">Requires Human Review</div>
            )}
          </div>
        </div>

        {/* Critical Findings */}
        {data.critical_findings && data.critical_findings.length > 0 && (
          <div className="data-section critical">
            <div className="section-header">
              <h3>Critical Findings</h3>
            </div>
            <div className="section-content">
              {data.critical_findings.map((finding, i) => (
                <div key={i} className="critical-item">{finding}</div>
              ))}
            </div>
          </div>
        )}

        {/* Raw Fields - Tabular Display (V2 format) */}
        {hasRawFields && (
          <div className="data-section">
            <div className="section-header">
              <h3>Extracted Fields</h3>
              <span className="field-count">{Object.keys(rawFields).length} fields</span>
            </div>
            <div className="section-content">
              <div className="table-container">
                <table className="fields-table">
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
                        <td className="field-value">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Extracted Values - Lab Results Style (Array format) */}
        {extractedValues.length > 0 && (
          <div className="data-section">
            <div className="section-header">
              <h3>Test Results / Measurements</h3>
              <span className="field-count">{extractedValues.length} items</span>
            </div>
            <div className="section-content">
              <div className="table-container">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Value</th>
                      <th>Unit</th>
                      <th>Reference</th>
                      <th>Flag</th>
                    </tr>
                  </thead>
                  <tbody>
                    {extractedValues.map((item, i) => (
                      <tr key={i} className={item.abnormal_flag ? 'abnormal' : ''}>
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
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Sections */}
        {data.sections && Object.entries(data.sections)
          .filter(([name]) => name !== 'extracted_fields') // Skip raw_fields, already shown above
          .map(([name, content]) => (
            <div key={name} className="data-section">
              <div className="section-header">
                <h3>{name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
              </div>
              <div className="section-content">
                {typeof content === 'string' ? (
                  <p>{content || 'No data'}</p>
                ) : Array.isArray(content) ? (
                  content.length > 0 ? (
                    typeof content[0] === 'object' ? (
                      <div className="table-container">
                        <table>
                          <thead>
                            <tr>
                              {Object.keys(content[0]).map(key => (
                                <th key={key}>{key.replace(/_/g, ' ')}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {content.map((item, i) => (
                              <tr key={i}>
                                {Object.values(item).map((val, j) => (
                                  <td key={j}>{String(val)}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <ul className="simple-list">
                        {content.map((item, i) => (
                          <li key={i}>{String(item)}</li>
                        ))}
                      </ul>
                    )
                  ) : <p>No data</p>
                ) : typeof content === 'object' ? (
                  <div className="object-view">
                    {Object.entries(content).map(([key, value]) => (
                      <div key={key} className="field">
                        <label>{key.replace(/_/g, ' ')}</label>
                        <span>{typeof value === 'object' ? JSON.stringify(value) : String(value) || '-'}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p>{String(content)}</p>
                )}
              </div>
            </div>
          ))}
      </div>
    );
  };

  const renderJsonView = () => (
    <div className="json-view">
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );

  const renderRawText = () => (
    <div className="raw-text-view">
      <pre>{data.raw_text || 'No raw text available'}</pre>
    </div>
  );

  return (
    <div className="extracted-data">
      <div className="tabs">
        <button
          className={activeTab === 'structured' ? 'active' : ''}
          onClick={() => setActiveTab('structured')}
        >
          Structured
        </button>
        <button
          className={activeTab === 'json' ? 'active' : ''}
          onClick={() => setActiveTab('json')}
        >
          JSON
        </button>
        <button
          className={activeTab === 'raw' ? 'active' : ''}
          onClick={() => setActiveTab('raw')}
        >
          Raw Text
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'structured' && renderStructuredView()}
        {activeTab === 'json' && renderJsonView()}
        {activeTab === 'raw' && renderRawText()}
      </div>
    </div>
  );
}

export default ExtractedData;
