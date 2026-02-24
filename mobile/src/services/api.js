// ============================================================================
// API Service for Mobile App
// ============================================================================

import axios from 'axios';
import * as FileSystem from 'expo-file-system';

// Use your machine's local IP for device/Expo Go testing
// localhost only works in web browser, not on physical devices
const API_BASE = 'http://192.168.2.20:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============================================================================
// Document Upload
// ============================================================================

/**
 * Upload a document image from the camera or gallery
 * @param {string} imageUri - Local URI of the image
 * @param {boolean} force - Force upload even if quality is poor
 * @returns {Promise<Object>} Upload result with file_id
 */
export const uploadDocument = async (imageUri, force = false) => {
  const formData = new FormData();

  // Get file info
  const fileInfo = await FileSystem.getInfoAsync(imageUri);
  const fileName = imageUri.split('/').pop();
  const fileType = fileName.endsWith('.png') ? 'image/png' : 'image/jpeg';

  formData.append('file', {
    uri: imageUri,
    name: fileName,
    type: fileType,
  });

  const url = force ? '/api/upload?force=true' : '/api/upload';

  const response = await api.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Upload and analyze document quality
 * @param {string} imageUri - Local URI of the image
 * @param {boolean} force - Force upload even if quality is poor
 * @returns {Promise<Object>} Upload result with quality analysis
 */
export const uploadAndAnalyze = async (imageUri, force = false) => {
  const formData = new FormData();

  const fileName = imageUri.split('/').pop();
  const fileType = fileName.endsWith('.png') ? 'image/png' : 'image/jpeg';

  formData.append('file', {
    uri: imageUri,
    name: fileName,
    type: fileType,
  });

  const url = force ? '/api/upload-and-analyze?force=true' : '/api/upload-and-analyze';

  const response = await api.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

// ============================================================================
// Document Processing
// ============================================================================

/**
 * Start processing an uploaded document
 * @param {string} fileId - File ID from upload
 * @param {string} documentType - Type of document (auto, lab, radiology, prescription)
 * @returns {Promise<Object>} Job info with job_id
 */
export const startProcessing = async (fileId, documentType = 'auto') => {
  const response = await api.post(
    `/api/v2/process?file_id=${fileId}&document_type=${documentType}`
  );
  return response.data;
};

/**
 * Get job status and results
 * @param {string} jobId - Job ID from processing
 * @returns {Promise<Object>} Job status and results
 */
export const getJobStatus = async (jobId) => {
  const response = await api.get(`/api/v2/jobs/${jobId}`);
  return response.data;
};

/**
 * Get workflow steps for a job
 * @param {string} jobId - Job ID
 * @returns {Promise<Object>} Workflow steps
 */
export const getWorkflowSteps = async (jobId) => {
  const response = await api.get(`/api/v2/jobs/${jobId}/workflow`);
  return response.data;
};

/**
 * List all jobs
 * @returns {Promise<Object>} List of jobs
 */
export const listJobs = async () => {
  const response = await api.get('/api/v2/jobs');
  return response.data;
};

/**
 * Delete a job
 * @param {string} jobId - Job ID to delete
 * @returns {Promise<Object>} Deletion confirmation
 */
export const deleteJob = async (jobId) => {
  const response = await api.delete(`/api/v2/jobs/${jobId}`);
  return response.data;
};

// ============================================================================
// Document Quality
// ============================================================================

/**
 * Analyze document quality
 * @param {string} fileId - File ID from upload
 * @returns {Promise<Object>} Quality analysis report
 */
export const analyzeQuality = async (fileId) => {
  const response = await api.get(`/api/document/${fileId}/quality`);
  return response.data;
};

/**
 * Get document info
 * @param {string} fileId - File ID
 * @returns {Promise<Object>} Document info
 */
export const getDocumentInfo = async (fileId) => {
  const response = await api.get(`/api/document/${fileId}/info`);
  return response.data;
};

// ============================================================================
// Health Check
// ============================================================================

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export const healthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

// ============================================================================
// Utilities
// ============================================================================

/**
 * Get document URL for viewing
 * @param {string} fileId - File ID
 * @returns {string} Document URL
 */
export const getDocumentUrl = (fileId) => {
  return `${API_BASE}/api/document/${fileId}`;
};

/**
 * Update API base URL (for connecting to different servers)
 * @param {string} url - New base URL
 */
export const setApiBaseUrl = (url) => {
  api.defaults.baseURL = url;
};

/**
 * Get current API base URL
 * @returns {string} Current base URL
 */
export const getApiBaseUrl = () => {
  return api.defaults.baseURL;
};

export default api;
