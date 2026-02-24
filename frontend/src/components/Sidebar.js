import React from 'react';

function Sidebar({
  samples,
  onFileSelect,
  selectedFile,
  documentType,
  onDocumentTypeChange,
  onProcess,
  isProcessing,
}) {
  // Group samples by category
  const groupedSamples = samples.reduce((acc, sample) => {
    const category = sample.category || 'other';
    if (!acc[category]) acc[category] = [];
    acc[category].push(sample);
    return acc;
  }, {});

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>Document Type</h2>
        <select
          className="document-type-select"
          value={documentType}
          onChange={(e) => onDocumentTypeChange(e.target.value)}
        >
          <option value="lab">Lab Report</option>
          <option value="radiology">Radiology Report</option>
          <option value="prescription">Prescription</option>
        </select>
      </div>

      <div className="sidebar-samples">
        <h2 style={{ fontSize: '14px', fontWeight: 600, color: '#a0a0c0', marginBottom: '12px' }}>
          Sample Documents
        </h2>
        {Object.entries(groupedSamples).map(([category, items]) => (
          <div key={category} className="sample-category">
            <h3>{category}</h3>
            {items.map((sample) => (
              <div
                key={sample.path}
                className={`sample-item ${selectedFile?.path === sample.path ? 'selected' : ''}`}
                onClick={() => onFileSelect(sample, sample.type)}
              >
                <div className="sample-item-name">{sample.name}</div>
              </div>
            ))}
          </div>
        ))}
        {samples.length === 0 && (
          <div style={{ color: '#6a6a8e', fontSize: '13px', textAlign: 'center', padding: '20px' }}>
            No sample documents found.
            <br />
            Add files to data/samples/
          </div>
        )}
      </div>

      <div className="sidebar-actions">
        <button
          className={`process-button ${isProcessing ? 'processing' : ''}`}
          onClick={onProcess}
          disabled={!selectedFile || isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Process Document'}
        </button>
      </div>
    </div>
  );
}

export default Sidebar;
