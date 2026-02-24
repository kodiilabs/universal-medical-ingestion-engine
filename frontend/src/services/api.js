// ============================================================================
// API Service Layer
// ============================================================================

import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// Document Upload & Processing
// ============================================================================

export const uploadDocument = async (file, force = false) => {
  const formData = new FormData();
  formData.append('file', file);

  const url = force ? '/api/upload?force=true' : '/api/upload';
  const response = await api.post(url, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const startProcessing = async (fileId, strategy = 'router', processingMode = 'auto') => {
  // Using v2 extraction-first pipeline
  const response = await api.post(
    `/api/v2/process?file_id=${fileId}&strategy=${strategy}&processing_mode=${processingMode}`
  );
  return response.data;
};

export const getJobStatus = async (jobId) => {
  const response = await api.get(`/api/v2/jobs/${jobId}`);
  return response.data;
};

export const getWorkflowSteps = async (jobId) => {
  const response = await api.get(`/api/v2/jobs/${jobId}/workflow`);
  return response.data;
};

export const listJobs = async () => {
  const response = await api.get('/api/v2/jobs');
  return response.data;
};

export const deleteJob = async (jobId) => {
  const response = await api.delete(`/api/v2/jobs/${jobId}`);
  return response.data;
};

export const deleteJobsBatch = async (jobIds) => {
  const response = await api.post('/api/v2/jobs/batch-delete', { job_ids: jobIds });
  return response.data;
};

export const reprocessJob = async (jobId, documentType = null) => {
  const params = new URLSearchParams();
  if (documentType) params.set('document_type', documentType);
  const qs = params.toString() ? `?${params.toString()}` : '';
  const response = await api.post(`/api/v2/jobs/${jobId}/reprocess${qs}`);
  return response.data;
};

// ============================================================================
// Sample Documents
// ============================================================================

export const listSamples = async () => {
  const response = await api.get('/api/samples');
  return response.data;
};

export const processSample = async (filePath, strategy = 'router') => {
  // Using v2 extraction-first pipeline
  const response = await api.post('/api/v2/samples/process', null, {
    params: { file_path: filePath, strategy: strategy }
  });
  return response.data;
};

// ============================================================================
// Document Retrieval
// ============================================================================

export const getDocumentUrl = (fileId) => {
  return `${API_BASE}/api/document/${fileId}`;
};

// ============================================================================
// Audit Trail
// ============================================================================

export const getAuditTrail = async (documentId) => {
  const response = await api.get(`/api/audit/${documentId}`);
  return response.data;
};

// ============================================================================
// Review Queue
// ============================================================================

export const getReviewQueue = async (priority = null) => {
  const params = priority ? { priority } : {};
  const response = await api.get('/api/review-queue', { params });
  return response.data;
};

export const submitReview = async (jobId, reviewData) => {
  const response = await api.post(`/api/jobs/${jobId}/review`, reviewData);
  return response.data;
};

// ============================================================================
// Templates
// ============================================================================

export const listTemplates = async () => {
  const response = await api.get('/api/templates');
  return response.data;
};

export const getTemplate = async (templateId) => {
  const response = await api.get(`/api/templates/${templateId}`);
  return response.data;
};

// ============================================================================
// Document Chat
// ============================================================================

export const sendChatMessage = async (jobId, message) => {
  const response = await api.post(`/api/v2/jobs/${jobId}/chat`, { message });
  return response.data;
};

export const getChatSuggestions = async (jobId) => {
  const response = await api.get(`/api/v2/jobs/${jobId}/chat/suggestions`);
  return response.data;
};

export const clearChatHistory = async (jobId) => {
  const response = await api.post(`/api/v2/jobs/${jobId}/chat`, {
    message: '',
    clear_history: true,
  });
  return response.data;
};

// ============================================================================
// Health Check
// ============================================================================

export const healthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export default api;
