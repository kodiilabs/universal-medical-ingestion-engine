// ============================================================================
// Upload Page - Drag-drop document upload
// ============================================================================

import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload as UploadIcon, FileText, X, Loader2 } from 'lucide-react';
import { uploadDocument, startProcessing } from '../../services/api';
import './Upload.css';

const DOCUMENT_TYPES = [
  { value: 'auto', label: 'Auto-detect' },
  { value: 'lab', label: 'Lab Report' },
  { value: 'radiology', label: 'Radiology Report' },
  { value: 'prescription', label: 'Prescription' },
  { value: 'pathology', label: 'Pathology Report' },
  { value: 'discharge_summary', label: 'Discharge Summary' },
  { value: 'operative_note', label: 'Operative Note' },
];

const PROCESSING_MODES = [
  { value: 'auto', label: 'Auto', description: 'App decides — uses OCR, adds VLM for low-quality scans' },
  { value: 'fast', label: 'Fast', description: 'OCR only — fastest, best for clear digital PDFs' },
  { value: 'accurate', label: 'Most Accurate', description: 'VLM for every page — slowest, best for handwritten/complex docs' },
];

const Upload = () => {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [documentType, setDocumentType] = useState('auto');
  const [processingMode, setProcessingMode] = useState('auto');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [qualityError, setQualityError] = useState(null); // For image quality issues

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && isValidFile(droppedFile)) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Please upload a PDF or image file');
    }
  }, []);

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && isValidFile(selectedFile)) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please upload a PDF or image file');
    }
  };

  const isValidFile = (file) => {
    const validTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
    return validTypes.includes(file.type);
  };

  const handleSubmit = async (forceUpload = false) => {
    if (!file) return;

    setUploading(true);
    setError(null);
    setQualityError(null);

    try {
      // Step 1: Upload the file (with force flag if retrying)
      const uploadResult = await uploadDocument(file, forceUpload);

      // Step 2: Start processing
      const processResult = await startProcessing(uploadResult.file_id, documentType, processingMode);

      // Step 3: Navigate to processing dashboard
      navigate(`/dashboard?job=${processResult.job_id}`);
    } catch (err) {
      const detail = err.response?.data?.detail;

      // Check if this is a quality error (object with error key)
      if (detail && typeof detail === 'object' && detail.error === 'image_quality_too_low') {
        setQualityError(detail);
        setError(null);
      } else {
        // Regular error - extract message string
        const errorMessage = typeof detail === 'string'
          ? detail
          : (detail?.message || 'Upload failed. Please try again.');
        setError(errorMessage);
        setQualityError(null);
      }
      setUploading(false);
    }
  };

  const removeFile = () => {
    setFile(null);
    setError(null);
    setQualityError(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-header">
          <h1>Upload Document</h1>
          <p>Upload a medical document for AI-powered extraction</p>
        </div>

        <div
          className={`dropzone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {!file ? (
            <>
              <div className="dropzone-icon">
                <UploadIcon size={48} />
              </div>
              <div className="dropzone-text">
                <p className="dropzone-primary">Drag & drop your document here</p>
                <p className="dropzone-secondary">or click to browse</p>
              </div>
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg"
                onChange={handleFileSelect}
                className="dropzone-input"
              />
              <div className="dropzone-formats">
                Supported: PDF, PNG, JPG
              </div>
            </>
          ) : (
            <div className="file-preview">
              <div className="file-icon">
                <FileText size={32} />
              </div>
              <div className="file-info">
                <p className="file-name">{file.name}</p>
                <p className="file-size">{formatFileSize(file.size)}</p>
              </div>
              <button className="file-remove" onClick={removeFile}>
                <X size={20} />
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="upload-error">
            {error}
          </div>
        )}

        {qualityError && (
          <div className="upload-quality-error">
            <div className="quality-error-header">
              <strong>⚠️ Image Quality Too Low</strong>
            </div>
            <p>{qualityError.message}</p>
            {qualityError.quality?.resolution && (
              <p className="quality-resolution">
                Current: {qualityError.quality.resolution.width}×{qualityError.quality.resolution.height} pixels
                (minimum: 600×450)
              </p>
            )}
            {qualityError.tips && qualityError.tips.length > 0 && (
              <div className="quality-tips">
                <strong>Tips for better results:</strong>
                <ul>
                  {qualityError.tips.map((tip, i) => (
                    <li key={i}>{tip}</li>
                  ))}
                </ul>
              </div>
            )}
            {qualityError.can_force && (
              <button
                className="btn btn-secondary quality-force-btn"
                onClick={() => handleSubmit(true)}
                disabled={uploading}
              >
                Try Anyway (not recommended)
              </button>
            )}
          </div>
        )}

        <div className="upload-options">
          <div className="option-group">
            <label>Document Type</label>
            <select
              value={documentType}
              onChange={(e) => setDocumentType(e.target.value)}
              className="input"
            >
              {DOCUMENT_TYPES.map(({ value, label }) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>
          <div className="option-group">
            <label>Processing Mode</label>
            <select
              value={processingMode}
              onChange={(e) => setProcessingMode(e.target.value)}
              className="input"
            >
              {PROCESSING_MODES.map(({ value, label }) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
            <span className="option-description">
              {PROCESSING_MODES.find(m => m.value === processingMode)?.description}
            </span>
          </div>
        </div>

        <button
          className="btn btn-primary upload-submit"
          onClick={handleSubmit}
          disabled={!file || uploading}
        >
          {uploading ? (
            <>
              <Loader2 size={20} className="spinning" />
              Processing...
            </>
          ) : (
            <>
              <UploadIcon size={20} />
              Process Document
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default Upload;
