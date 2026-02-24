// ============================================================================
// Processing Dashboard - Live job monitoring with workflow steps
// ============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  CheckCircle,
  Circle,
  Loader2,
  XCircle,
  FileText,
  Clock,
  ArrowRight
} from 'lucide-react';
import { getJobStatus, listJobs } from '../../services/api';
import { StatusBadge, ConfidenceBadge } from '../../components/shared';
import './Dashboard.css';

const POLL_INTERVAL = 2000; // 2 seconds

const WorkflowStep = ({ step }) => {
  const getIcon = () => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle size={20} className="step-icon completed" />;
      case 'running':
        return <Loader2 size={20} className="step-icon running spinning" />;
      case 'failed':
        return <XCircle size={20} className="step-icon failed" />;
      default:
        return <Circle size={20} className="step-icon pending" />;
    }
  };

  return (
    <div className={`workflow-step ${step.status}`}>
      {getIcon()}
      <div className="step-content">
        <span className="step-name">{step.name}</span>
        {step.started_at && (
          <span className="step-time">
            {new Date(step.started_at).toLocaleTimeString()}
          </span>
        )}
      </div>
    </div>
  );
};

const JobCard = ({ job, isActive, onClick }) => {
  const getStatusClass = () => {
    if (job.requires_review) return 'review';
    return job.status;
  };

  return (
    <div
      className={`job-card ${isActive ? 'active' : ''} ${getStatusClass()}`}
      onClick={onClick}
    >
      <div className="job-card-header">
        <div className="job-type">
          <FileText size={16} />
          <span>{job.document_type || 'auto'}</span>
        </div>
        <StatusBadge status={job.requires_review ? 'review_required' : job.status} />
      </div>

      <div className="job-card-body">
        <p className="job-filename">{job.file_name}</p>
        <div className="job-meta">
          <Clock size={14} />
          <span>{new Date(job.created_at).toLocaleString()}</span>
        </div>
      </div>

      {job.result?.confidence && (
        <div className="job-card-footer">
          <ConfidenceBadge score={job.result.confidence} size="sm" />
        </div>
      )}
    </div>
  );
};

const Dashboard = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const activeJobId = searchParams.get('job');

  const [jobs, setJobs] = useState([]);
  const [activeJob, setActiveJob] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch all jobs
  const fetchJobs = useCallback(async () => {
    try {
      const response = await listJobs();
      setJobs(response.jobs || []);
    } catch (err) {
      console.error('Failed to fetch jobs:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch active job details
  const fetchActiveJob = useCallback(async () => {
    if (!activeJobId) return;

    try {
      const job = await getJobStatus(activeJobId);
      setActiveJob(job);

      // Auto-navigate when complete
      if (job.status === 'completed' || job.status === 'failed') {
        // Update jobs list
        fetchJobs();
      }
    } catch (err) {
      console.error('Failed to fetch job:', err);
    }
  }, [activeJobId, fetchJobs]);

  // Initial load
  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Poll for active job
  useEffect(() => {
    if (!activeJobId) return;

    fetchActiveJob();

    const interval = setInterval(() => {
      if (activeJob?.status === 'pending' || activeJob?.status === 'processing') {
        fetchActiveJob();
      }
    }, POLL_INTERVAL);

    return () => clearInterval(interval);
  }, [activeJobId, activeJob?.status, fetchActiveJob]);

  const handleJobClick = (job) => {
    if (job.status === 'completed') {
      navigate(`/documents/${job.job_id}`);
    } else {
      navigate(`/dashboard?job=${job.job_id}`);
    }
  };

  const handleViewDetails = () => {
    if (activeJob?.status === 'completed') {
      navigate(`/documents/${activeJob.job_id}`);
    }
  };

  if (loading) {
    return (
      <div className="dashboard-loading">
        <Loader2 size={32} className="spinning" />
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Processing Dashboard</h1>
        <p>Monitor document processing in real-time</p>
      </div>

      <div className="dashboard-content">
        {/* Jobs List */}
        <div className="jobs-panel">
          <div className="panel-header">
            <h3>Recent Jobs</h3>
            <span className="job-count">{jobs.length}</span>
          </div>
          <div className="jobs-list">
            {jobs.length === 0 ? (
              <div className="empty-state">
                <FileText size={32} />
                <p>No documents processed yet</p>
              </div>
            ) : (
              jobs.map((job) => (
                <JobCard
                  key={job.job_id}
                  job={job}
                  isActive={job.job_id === activeJobId}
                  onClick={() => handleJobClick(job)}
                />
              ))
            )}
          </div>
        </div>

        {/* Active Job Details */}
        <div className="details-panel">
          {activeJob ? (
            <>
              <div className="panel-header">
                <h3>Processing Status</h3>
                <StatusBadge status={activeJob.status} />
              </div>

              <div className="job-details">
                <div className="detail-row">
                  <span className="detail-label">File</span>
                  <span className="detail-value">{activeJob.file_name}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Type</span>
                  <span className="detail-value">{activeJob.document_type}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Started</span>
                  <span className="detail-value">
                    {new Date(activeJob.created_at).toLocaleString()}
                  </span>
                </div>
                {activeJob.completed_at && (
                  <div className="detail-row">
                    <span className="detail-label">Completed</span>
                    <span className="detail-value">
                      {new Date(activeJob.completed_at).toLocaleString()}
                    </span>
                  </div>
                )}
              </div>

              {/* Workflow Steps */}
              <div className="workflow-section">
                <h4>Workflow Progress</h4>
                <div className="workflow-steps">
                  {(activeJob.workflow_steps || []).map((step, index) => (
                    <WorkflowStep key={step.id || index} step={step} />
                  ))}
                </div>
              </div>

              {/* Result Preview */}
              {activeJob.result && (
                <div className="result-preview">
                  <h4>Result</h4>
                  <div className="result-summary">
                    {activeJob.result.confidence && (
                      <div className="result-item">
                        <span>Confidence</span>
                        <ConfidenceBadge score={activeJob.result.confidence} />
                      </div>
                    )}
                    {activeJob.result.requires_review && (
                      <div className="result-item warning">
                        <span>Requires Review</span>
                      </div>
                    )}
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={handleViewDetails}
                  >
                    View Details
                    <ArrowRight size={16} />
                  </button>
                </div>
              )}

              {/* Error */}
              {activeJob.error && (
                <div className="job-error">
                  <XCircle size={20} />
                  <span>{activeJob.error}</span>
                </div>
              )}
            </>
          ) : (
            <div className="empty-state">
              <FileText size={48} />
              <h3>No job selected</h3>
              <p>Select a job from the list or upload a new document</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
