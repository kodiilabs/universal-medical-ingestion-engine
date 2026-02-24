import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export async function listSamples() {
  const response = await api.get('/samples');
  return response.data;
}

export async function processDocument(filePath, documentType) {
  const response = await api.post('/samples/process', null, {
    params: {
      file_path: filePath,
      document_type: documentType,
    },
  });
  return response.data;
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function startProcessing(fileId, documentType) {
  const response = await api.post('/process', null, {
    params: {
      file_id: fileId,
      document_type: documentType,
    },
  });
  return response.data;
}

export async function getJobStatus(jobId) {
  const response = await api.get(`/jobs/${jobId}`);
  return response.data;
}

export async function getWorkflowSteps(jobId) {
  const response = await api.get(`/jobs/${jobId}/workflow`);
  return response.data;
}

export async function listJobs() {
  const response = await api.get('/jobs');
  return response.data;
}

export async function deleteJob(jobId) {
  const response = await api.delete(`/jobs/${jobId}`);
  return response.data;
}

export function getDocumentUrl(fileId) {
  return `${API_BASE}/document/${fileId}`;
}
