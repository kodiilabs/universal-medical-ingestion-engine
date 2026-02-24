// ============================================================================
// Process Flow Page - React Flow visualization of agentic document processing
// Enhanced to show detailed pipeline stages, parallel operations, and models
// ============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Handle,
  Position,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  ArrowLeft,
  Loader2,
  Play,
  GitBranch,
  FileText,
  CheckCircle,
  AlertCircle,
  Database,
  Cpu,
  Filter,
  Zap,
  Brain,
  Search,
  Shield,
  Clock,
  Beaker,
  MessageSquare,
  Layers,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Info,
  Eye,
  Type,
  SplitSquareHorizontal,
  Merge,
  Box,
  Settings,
  Sparkles,
  Cloud
} from 'lucide-react';
import { getJobStatus } from '../../services/api';
import './ProcessFlow.css';

// ============================================================================
// Icon and Color Helpers
// ============================================================================

const getStageIcon = (name) => {
  const lower = (name || '').toLowerCase();
  if (lower.includes('upload') || lower.includes('input')) return <FileText size={16} />;
  if (lower === 'document_intake') return <FileText size={16} />;
  if (lower === 'cloud_extraction') return <Cloud size={16} />;
  if (lower === 'data_structuring') return <Layers size={16} />;
  if (lower.includes('text_extraction') || lower.includes('ocr')) return <Type size={16} />;
  if (lower.includes('vlm') || lower.includes('vision')) return <Eye size={16} />;
  if (lower.includes('consensus')) return <Merge size={16} />;
  if (lower.includes('raw_field') || lower.includes('pre-extract')) return <Search size={16} />;
  if (lower.includes('similar') || lower.includes('lookup')) return <Database size={16} />;
  if (lower.includes('parallel') || lower.includes('extraction')) return <Cpu size={16} />;
  if (lower.includes('classif')) return <Filter size={16} />;
  if (lower.includes('database_validation') || lower.includes('db_valid')) return <Shield size={16} />;
  if (lower.includes('vector')) return <Box size={16} />;
  if (lower.includes('enrich')) return <Sparkles size={16} />;
  if (lower.includes('llm') || lower.includes('ai')) return <Brain size={16} />;
  return <Settings size={16} />;
};

const getStageColor = (name, status) => {
  if (status === 'failed') return '#EF4444';
  if (status === 'running') return '#3B82F6';
  if (status === 'skipped') return '#9CA3AF';

  const lower = (name || '').toLowerCase();
  // Cloud path stages
  if (lower === 'document_intake') return '#8B5CF6';  // Purple
  if (lower === 'cloud_extraction') return '#0EA5E9'; // Sky blue
  // Local path stages
  if (lower.includes('text_extraction')) return '#8B5CF6';  // Purple
  if (lower.includes('vlm') || lower.includes('vision')) return '#EC4899'; // Pink
  if (lower.includes('consensus')) return '#6366F1';  // Indigo
  if (lower.includes('ocr') || lower.includes('paddle')) return '#F59E0B'; // Amber
  if (lower.includes('raw_field')) return '#14B8A6';  // Teal
  if (lower.includes('similar')) return '#06B6D4';    // Cyan
  if (lower.includes('parallel') || lower.includes('llm')) return '#10B981'; // Emerald
  if (lower.includes('classif')) return '#3B82F6';    // Blue
  if (lower.includes('database_validation')) return '#059669'; // Green (verification)
  if (lower.includes('vector')) return '#8B5CF6';     // Purple
  if (lower.includes('enrich')) return '#F97316';     // Orange
  return '#6B7280';
};

// ============================================================================
// Custom Node Components
// ============================================================================

// Start node - Document input
const StartNode = ({ data }) => (
  <div className={`flow-node start-node ${data.status}`}>
    <Handle type="source" position={Position.Bottom} />
    <div className="node-icon start-icon">
      <Play size={20} />
    </div>
    <div className="node-content">
      <span className="node-label">{data.label}</span>
      {data.subtitle && <span className="node-subtitle">{data.subtitle}</span>}
      {data.fileType && <span className="node-filetype">{data.fileType}</span>}
    </div>
  </div>
);

// Pipeline Stage Node - Main processing stages with details
const PipelineStageNode = ({ data }) => {
  const [expanded, setExpanded] = useState(false);
  const stageColor = getStageColor(data.stageName || data.label, data.status);

  return (
    <div
      className={`flow-node pipeline-stage-node ${data.status}`}
      style={{ borderLeftColor: stageColor }}
    >
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />

      <div className="stage-header" onClick={() => setExpanded(!expanded)}>
        <div className="stage-icon-wrapper" style={{ backgroundColor: `${stageColor}15`, color: stageColor }}>
          {getStageIcon(data.stageName || data.label)}
        </div>
        <div className="stage-info">
          <span className="stage-name">{data.label}</span>
          <div className="stage-meta">
            <span className={`stage-status-badge ${data.status}`}>{data.status}</span>
            {data.duration && (
              <span className="stage-duration">
                {data.duration < 1 ? `${Math.round(data.duration * 1000)}ms` : `${data.duration.toFixed(1)}s`}
              </span>
            )}
          </div>
        </div>
        {data.details && Object.keys(data.details).length > 0 && (
          <div className="stage-expand-btn">
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>
        )}
      </div>

      {/* Description */}
      {data.description && (
        <div className="stage-description">{data.description}</div>
      )}

      {/* Model badges */}
      {(data.vlmModel || data.llmModel || data.embeddingModel) && (
        <div className="stage-models">
          {data.vlmModel && (
            <span className="model-badge vlm">
              <Eye size={10} />
              {data.vlmModel}
            </span>
          )}
          {data.llmModel && (
            <span className="model-badge llm">
              <Brain size={10} />
              {data.llmModel}
            </span>
          )}
          {data.embeddingModel && (
            <span className="model-badge embedding">
              <Box size={10} />
              {data.embeddingModel}
            </span>
          )}
        </div>
      )}

      {/* Expanded details */}
      {expanded && data.details && (
        <div className="stage-details">
          {Object.entries(data.details)
            .filter(([key, value]) => !['description', 'vlm_model', 'llm_model', 'embedding_model'].includes(key) && value != null)
            .slice(0, 8)
            .map(([key, value]) => (
              <div key={key} className="detail-row">
                <span className="detail-label">{key.replace(/_/g, ' ')}:</span>
                <span className="detail-value">
                  {typeof value === 'boolean' ? (value ? 'Yes' : 'No') :
                   typeof value === 'number' && key.includes('confidence') ? `${Math.round(value * 100)}%` :
                   typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </span>
              </div>
            ))}
        </div>
      )}

      {/* Running spinner with per-page progress */}
      {data.status === 'running' && (
        <div className="stage-running">
          <Loader2 size={14} className="spinning" />
          <span>
            {data.details?.current_page && data.details?.total_pages
              ? `Extracting page ${data.details.current_page} of ${data.details.total_pages}...`
              : 'Processing...'}
          </span>
        </div>
      )}
    </div>
  );
};

// Consensus Node - Shows parallel VLM + OCR with merge
const ConsensusNode = ({ data }) => {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className={`flow-node consensus-node ${data.status}`}>
      <Handle type="target" position={Position.Top} />
      <Handle type="source" position={Position.Bottom} />

      <div className="consensus-header" onClick={() => setExpanded(!expanded)}>
        <div className="consensus-icon">
          <SplitSquareHorizontal size={18} />
        </div>
        <div className="consensus-title">
          <span className="consensus-name">{data.label}</span>
          <span className="consensus-subtitle">Parallel Extraction</span>
        </div>
        <span className={`consensus-status ${data.status}`}>{data.status}</span>
      </div>

      {expanded && (
        <div className="consensus-body">
          {/* VLM Branch */}
          <div className={`consensus-branch vlm ${data.vlmStatus || 'completed'}`}>
            <div className="branch-header">
              <Eye size={14} />
              <span>VLM</span>
              {data.vlmContribution && (
                <span className="branch-contribution">{Math.round(data.vlmContribution * 100)}%</span>
              )}
            </div>
            {data.vlmModel && (
              <div className="branch-model">{data.vlmModel}</div>
            )}
            {data.vlmConfidence && (
              <div className="branch-confidence">
                Confidence: {Math.round(data.vlmConfidence * 100)}%
              </div>
            )}
          </div>

          {/* Merge indicator */}
          <div className="consensus-merge">
            <Merge size={16} />
            <span>Consensus</span>
          </div>

          {/* OCR Branch */}
          <div className={`consensus-branch ocr ${data.ocrStatus || 'completed'}`}>
            <div className="branch-header">
              <Type size={14} />
              <span>OCR</span>
              {data.ocrContribution && (
                <span className="branch-contribution">{Math.round(data.ocrContribution * 100)}%</span>
              )}
            </div>
            {data.ocrEngine && (
              <div className="branch-model">{data.ocrEngine}</div>
            )}
            {data.ocrConfidence && (
              <div className="branch-confidence">
                Confidence: {Math.round(data.ocrConfidence * 100)}%
              </div>
            )}
          </div>
        </div>
      )}

      {/* Result summary */}
      {data.primarySource && (
        <div className="consensus-result">
          <span className="result-label">Primary source:</span>
          <span className={`result-value ${data.primarySource}`}>{data.primarySource}</span>
        </div>
      )}
    </div>
  );
};

// Parallel Group Node - Shows operations running in parallel
const ParallelGroupNode = ({ data }) => (
  <div className={`flow-node parallel-group-node ${data.status}`}>
    <Handle type="target" position={Position.Top} />
    <Handle type="source" position={Position.Bottom} />
    <div className="parallel-header">
      <Layers size={16} />
      <span>{data.label}</span>
      <span className="parallel-indicator">PARALLEL</span>
    </div>
    <div className="parallel-items">
      {data.items?.map((item, idx) => (
        <div key={idx} className="parallel-item" style={{ borderColor: getStageColor(item.name) }}>
          <div className="parallel-item-icon" style={{ color: getStageColor(item.name) }}>
            {getStageIcon(item.name)}
          </div>
          <div className="parallel-item-info">
            <span className="parallel-item-name">{item.name}</span>
            {item.model && <span className="parallel-item-model">{item.model}</span>}
          </div>
          <span className={`parallel-item-status ${item.status || 'completed'}`}>
            {item.status === 'completed' ? <CheckCircle size={12} /> :
             item.status === 'failed' ? <AlertCircle size={12} /> :
             <Loader2 size={12} className="spinning" />}
          </span>
        </div>
      ))}
    </div>
  </div>
);

// End node - Final result
const EndNode = ({ data }) => (
  <div className={`flow-node end-node ${data.status}`}>
    <Handle type="target" position={Position.Top} />
    <div className="end-icon">
      {data.status === 'completed' ? (
        <CheckCircle size={24} />
      ) : data.status === 'failed' ? (
        <AlertCircle size={24} />
      ) : (
        <Database size={24} />
      )}
    </div>
    <div className="end-content">
      <span className="end-label">{data.label}</span>
      {data.confidence > 0 && (
        <div className="end-confidence">
          <span className="confidence-label">Confidence</span>
          <span className="confidence-value">{Math.round(data.confidence * 100)}%</span>
        </div>
      )}
      {data.stats && (
        <div className="end-stats">
          {data.stats.map((stat, idx) => (
            <div key={idx} className="end-stat">
              <span className="stat-value">{stat.value}</span>
              <span className="stat-label">{stat.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
);

// Review node - Human review required
const ReviewNode = ({ data }) => (
  <div className="flow-node review-node">
    <Handle type="target" position={Position.Left} />
    <div className="review-icon">
      <AlertTriangle size={20} />
    </div>
    <div className="review-content">
      <span className="review-label">{data.label}</span>
      {data.reasons && (
        <div className="review-reasons">
          {data.reasons.map((reason, idx) => (
            <span key={idx} className="review-reason">{reason}</span>
          ))}
        </div>
      )}
    </div>
  </div>
);

const nodeTypes = {
  start: StartNode,
  pipelineStage: PipelineStageNode,
  consensus: ConsensusNode,
  parallelGroup: ParallelGroupNode,
  end: EndNode,
  review: ReviewNode,
};

// ============================================================================
// Flow Generation from workflow_steps
// ============================================================================

const generateFlowFromJob = (job) => {
  if (!job) return { nodes: [], edges: [] };

  const nodes = [];
  const edges = [];
  const result = job.result || {};
  const steps = job.workflow_steps || result.pipeline_stages || [];
  const textResult = result.text_result || {};

  let yPos = 0;
  const xCenter = 350;
  const nodeSpacing = 200;

  // Detect cloud vs local path
  const isCloudPath = steps.some(s => s.name === 'cloud_extraction');

  // Check if consensus extraction was used (local path only)
  const isConsensus = !isCloudPath && (textResult.consensus_used ||
    (textResult.extraction_method && textResult.extraction_method.includes('consensus')));

  // =========================================================================
  // START NODE
  // =========================================================================
  nodes.push({
    id: 'start',
    type: 'start',
    position: { x: xCenter, y: yPos },
    data: {
      label: 'Document Input',
      subtitle: job.file_name,
      fileType: job.file_name?.split('.').pop()?.toUpperCase(),
      status: 'completed'
    }
  });
  yPos += nodeSpacing;

  let lastNodeId = 'start';

  // =========================================================================
  // CLOUD PATH — GPT-4o Vision Pipeline
  // =========================================================================
  if (isCloudPath) {
    // --- Document Intake ---
    const intakeStep = steps.find(s => s.name === 'document_intake');
    if (intakeStep) {
      const details = intakeStep.details || {};
      nodes.push({
        id: 'document-intake',
        type: 'pipelineStage',
        position: { x: xCenter, y: yPos },
        data: {
          label: 'Document Intake',
          stageName: 'document_intake',
          status: intakeStep.status,
          duration: intakeStep.duration_seconds,
          description: details.description || `Analyzing ${details.file_type || ''} document`,
          details: {
            file_type: details.file_type,
            file_size_kb: details.file_size_kb,
            pages: details.pages,
            target_api: details.target_api,
            deployment: details.deployment,
          }
        }
      });
      edges.push({ id: 'start-intake', source: 'start', target: 'document-intake', type: 'smoothstep' });
      lastNodeId = 'document-intake';
      yPos += nodeSpacing;
    }

    // --- Cloud Extraction (GPT-4o API call) ---
    const cloudStep = steps.find(s => s.name === 'cloud_extraction');
    if (cloudStep) {
      const details = cloudStep.details || {};
      nodes.push({
        id: 'cloud-extraction',
        type: 'pipelineStage',
        position: { x: xCenter, y: yPos },
        data: {
          label: 'GPT-4o Vision API Call',
          stageName: 'cloud_extraction',
          status: cloudStep.status,
          duration: cloudStep.duration_seconds,
          description: details.description || 'Azure GPT-4o vision API call',
          llmModel: details.deployment || 'gpt-4o',
          details: {
            deployment: details.deployment,
            pages_processed: details.pages_processed,
            conversion_time: details.conversion_time,
            api_call_time: details.api_call_time,
            max_tokens: details.max_tokens,
            temperature: details.temperature,
          }
        }
      });
      edges.push({ id: `${lastNodeId}-cloud`, source: lastNodeId, target: 'cloud-extraction', type: 'smoothstep' });
      lastNodeId = 'cloud-extraction';
      yPos += nodeSpacing;
    }

    // --- Extraction + Classification (always separate nodes) ---
    const cloudParallelStep = steps.find(s => s.name === 'parallel_extraction');
    if (cloudParallelStep) {
      const details = cloudParallelStep.details || {};

      // Build extraction description
      const extractionParts = [];
      if (details.test_results_count > 0) extractionParts.push(`${details.test_results_count} tests`);
      if (details.medications_count > 0) extractionParts.push(`${details.medications_count} meds`);
      if (details.findings_count > 0) extractionParts.push(`${details.findings_count} findings`);
      if (details.procedures_count > 0) extractionParts.push(`${details.procedures_count} procedures`);
      const extractionDesc = extractionParts.length > 0
        ? `Structured ${extractionParts.join(', ')}` : 'Structuring GPT-4o response';

      // Build classification description
      const classType = details.document_type || 'unknown';
      const classConf = details.classification_confidence
        ? `${Math.round(details.classification_confidence * 100)}%`
        : '';
      const classDesc = classConf ? `${classType} (${classConf} confidence)` : classType;

      if (details.parallel) {
        // PARALLEL: two nodes side by side with fork-join edges
        const xOffset = 220;

        nodes.push({
          id: 'cloud-data-structuring',
          type: 'pipelineStage',
          position: { x: xCenter - xOffset, y: yPos },
          data: {
            label: 'Data Structuring',
            stageName: 'data_structuring',
            status: details.extraction_success ? 'completed' : 'failed',
            duration: cloudParallelStep.duration_seconds,
            description: extractionDesc,
            llmModel: details.deployment || 'gpt-4o',
            details: {
              test_results_count: details.test_results_count,
              medications_count: details.medications_count,
              findings_count: details.findings_count,
              procedures_count: details.procedures_count,
              patient_extracted: details.patient_extracted,
            }
          }
        });

        nodes.push({
          id: 'cloud-classification',
          type: 'pipelineStage',
          position: { x: xCenter + xOffset, y: yPos },
          data: {
            label: 'Document Classification',
            stageName: 'classification',
            status: details.classification_success ? 'completed' : 'failed',
            duration: cloudParallelStep.duration_seconds,
            description: classDesc,
            llmModel: details.deployment || 'gpt-4o',
            details: {
              document_type: details.document_type,
              classification_confidence: details.classification_confidence,
            }
          }
        });

        // Fork: previous node → both
        edges.push({ id: `${lastNodeId}-structuring`, source: lastNodeId, target: 'cloud-data-structuring', type: 'smoothstep' });
        edges.push({ id: `${lastNodeId}-classification`, source: lastNodeId, target: 'cloud-classification', type: 'smoothstep' });

        // Track both for join
        lastNodeId = 'cloud-parallel-join';
        yPos += nodeSpacing;
      } else {
        // SEQUENTIAL: two stacked nodes
        nodes.push({
          id: 'cloud-data-structuring',
          type: 'pipelineStage',
          position: { x: xCenter, y: yPos },
          data: {
            label: 'Data Structuring',
            stageName: 'data_structuring',
            status: details.extraction_success !== false ? cloudParallelStep.status : 'failed',
            duration: cloudParallelStep.duration_seconds,
            description: extractionDesc,
            llmModel: details.deployment || 'gpt-4o',
            details: {
              test_results_count: details.test_results_count,
              medications_count: details.medications_count,
              findings_count: details.findings_count,
              procedures_count: details.procedures_count,
              patient_extracted: details.patient_extracted,
            }
          }
        });
        edges.push({ id: `${lastNodeId}-structuring`, source: lastNodeId, target: 'cloud-data-structuring', type: 'smoothstep' });
        yPos += nodeSpacing;

        nodes.push({
          id: 'cloud-classification',
          type: 'pipelineStage',
          position: { x: xCenter, y: yPos },
          data: {
            label: 'Document Classification',
            stageName: 'classification',
            status: details.classification_success !== false ? cloudParallelStep.status : 'failed',
            description: classDesc,
            llmModel: details.deployment || 'gpt-4o',
            details: {
              document_type: details.document_type,
              classification_confidence: details.classification_confidence,
            }
          }
        });
        edges.push({ id: 'structuring-classification', source: 'cloud-data-structuring', target: 'cloud-classification', type: 'smoothstep' });

        lastNodeId = 'cloud-classification';
        yPos += nodeSpacing;
      }
    }

    // Helper: connect from cloud parallel fork or single lastNodeId
    const addCloudEdgesToTarget = (targetId) => {
      if (lastNodeId === 'cloud-parallel-join') {
        edges.push({ id: `cloud-structuring-${targetId}`, source: 'cloud-data-structuring', target: targetId, type: 'smoothstep' });
        edges.push({ id: `cloud-classification-${targetId}`, source: 'cloud-classification', target: targetId, type: 'smoothstep' });
      } else {
        edges.push({ id: `${lastNodeId}-${targetId}`, source: lastNodeId, target: targetId, type: 'smoothstep' });
      }
    };

    // --- Enrichment (cloud path, same as local) ---
    const enrichmentStep = steps.find(s => s.name.includes('enrichment'));
    if (enrichmentStep) {
      const details = enrichmentStep.details || {};
      nodes.push({
        id: 'enrichment',
        type: 'pipelineStage',
        position: { x: xCenter, y: yPos },
        data: {
          label: enrichmentStep.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          stageName: enrichmentStep.name,
          status: enrichmentStep.status,
          duration: enrichmentStep.duration_seconds,
          description: details.description || (enrichmentStep.status === 'skipped' ? details.reason : 'Type-specific enrichment'),
          llmModel: details.llm_model,
          details: details
        }
      });
      addCloudEdgesToTarget('enrichment');
      lastNodeId = 'enrichment';
      yPos += nodeSpacing;
    }

    // --- Database Validation (cloud path) ---
    const cloudDbValidationStep = steps.find(s => s.name === 'database_validation');
    if (cloudDbValidationStep) {
      const details = cloudDbValidationStep.details || {};
      const verified = details.verified_count || 0;
      const unverified = details.unverified_count || 0;
      const strengthMismatch = details.strength_mismatch_count || 0;
      const total = details.total_items_checked || 0;
      const dbName = details.database || 'Medical DB';

      let description = details.description || 'Validated against medical coding database';
      if (total > 0) {
        description = `${dbName}: ${verified}/${total} verified`;
        if (strengthMismatch > 0) description += ` — ${strengthMismatch} strength mismatch`;
        if (unverified > 0) description += ` — ${unverified} unverified`;
      }

      nodes.push({
        id: 'db-validation',
        type: 'pipelineStage',
        position: { x: xCenter, y: yPos },
        data: {
          label: `${dbName} Validation`,
          stageName: 'database_validation',
          status: cloudDbValidationStep.status,
          duration: cloudDbValidationStep.duration_seconds,
          description: description,
          warning: details.warning,
          details: {
            database: details.database,
            total_items_checked: total,
            verified_count: verified,
            unverified_count: unverified,
            strength_mismatch_count: strengthMismatch,
            rxnorm_codes_found: details.rxnorm_codes_found,
            loinc_codes_found: details.loinc_codes_found,
            ocr_corrections: details.ocr_corrections,
            interactions_detected: details.interactions_detected,
          }
        }
      });
      addCloudEdgesToTarget('db-validation');
      lastNodeId = 'db-validation';
      yPos += nodeSpacing;
    }

    // --- End Node (cloud path) ---
    yPos += 20;
    const endStats = [];
    if (result.extracted_values?.length > 0) {
      endStats.push({ value: result.extracted_values.length, label: 'values' });
    }
    if (result.raw_fields && Object.keys(result.raw_fields).length > 0) {
      endStats.push({ value: Object.keys(result.raw_fields).length, label: 'fields' });
    }
    if (result.total_time) {
      endStats.push({ value: `${result.total_time.toFixed(1)}s`, label: 'total' });
    }

    nodes.push({
      id: 'end',
      type: 'end',
      position: { x: xCenter, y: yPos },
      data: {
        label: job.status === 'completed' ? 'Processing Complete' :
               job.status === 'failed' ? 'Processing Failed' : 'Output',
        status: job.status,
        confidence: result.confidence,
        stats: endStats
      }
    });
    addCloudEdgesToTarget('end');

    // Review node (cloud path)
    if (result.requires_review) {
      nodes.push({
        id: 'review',
        type: 'review',
        position: { x: xCenter + 300, y: yPos },
        data: {
          label: 'Human Review Required',
          reasons: result.review_reasons || ['Low confidence']
        }
      });
      edges.push({
        id: 'end-review',
        source: 'end',
        target: 'review',
        type: 'smoothstep',
        style: { stroke: '#F59E0B', strokeDasharray: '5,5' },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#F59E0B' },
      });
    }

    return { nodes, edges };
  }

  // =========================================================================
  // LOCAL PATH — Standard extraction with local VLM/OCR
  // =========================================================================

  // =========================================================================
  // TEXT EXTRACTION - Show consensus if used
  // =========================================================================
  const textExtractionStep = steps.find(s => s.name === 'text_extraction');

  if (isConsensus) {
    // Show consensus extraction node
    const consensusDetails = textExtractionStep?.details || {};

    nodes.push({
      id: 'consensus-extraction',
      type: 'consensus',
      position: { x: xCenter, y: yPos },
      data: {
        label: 'Consensus Text Extraction',
        status: textExtractionStep?.status || 'completed',
        vlmModel: consensusDetails.vlm_model,
        vlmConfidence: parseFloat(textResult.extraction_method?.match(/vlm:(\d+)%/)?.[1] || 0) / 100,
        vlmContribution: textResult.extraction_method?.includes('vlm:100%') ? 1 : 0.5,
        vlmStatus: consensusDetails.vlm_used ? 'completed' : 'skipped',
        ocrEngine: consensusDetails.ocr_engine || 'PaddleOCR',
        ocrConfidence: textResult.confidence || 0.7,
        ocrContribution: textResult.extraction_method?.includes('ocr:') ?
          parseFloat(textResult.extraction_method.match(/ocr:(\d+)%/)?.[1] || 0) / 100 : 0,
        ocrStatus: consensusDetails.ocr_used ? 'completed' : 'skipped',
        primarySource: consensusDetails.method?.includes('vlm') ? 'vlm' :
                       consensusDetails.method?.includes('ocr') ? 'ocr' : 'consensus'
      }
    });

    edges.push({
      id: 'start-consensus',
      source: 'start',
      target: 'consensus-extraction',
      type: 'smoothstep',
    });

    lastNodeId = 'consensus-extraction';
    yPos += nodeSpacing + 60;

  } else if (textExtractionStep) {
    // Standard text extraction node
    const details = textExtractionStep.details || {};

    nodes.push({
      id: 'text-extraction',
      type: 'pipelineStage',
      position: { x: xCenter, y: yPos },
      data: {
        label: 'Text Extraction',
        stageName: 'text_extraction',
        status: textExtractionStep.status,
        duration: textExtractionStep.duration_seconds,
        description: details.description || `Extracted ${details.chars_extracted || 0} characters from ${details.pages || 1} page(s)`,
        vlmModel: details.vlm_model,
        details: {
          method: details.method,
          source_type: details.source_type,
          chars_extracted: details.chars_extracted,
          pages: details.pages,
        }
      }
    });

    edges.push({
      id: 'start-text',
      source: 'start',
      target: 'text-extraction',
      type: 'smoothstep',
    });

    lastNodeId = 'text-extraction';
    yPos += nodeSpacing;
  }

  // =========================================================================
  // FIELD PRE-EXTRACTION
  // =========================================================================
  const rawFieldStep = steps.find(s => s.name === 'raw_field_extraction');
  if (rawFieldStep) {
    const details = rawFieldStep.details || {};

    nodes.push({
      id: 'raw-field',
      type: 'pipelineStage',
      position: { x: xCenter, y: yPos },
      data: {
        label: 'Field Pre-Extraction',
        stageName: 'raw_field_extraction',
        status: rawFieldStep.status,
        duration: rawFieldStep.duration_seconds,
        description: details.description || `Pre-extracted ${details.fields_extracted || 0} key-value pairs`,
        details: {
          method: details.method,
          fields_extracted: details.fields_extracted,
          field_names: details.field_names?.slice(0, 5)?.join(', '),
        }
      }
    });

    edges.push({
      id: `${lastNodeId}-raw-field`,
      source: lastNodeId,
      target: 'raw-field',
      type: 'smoothstep',
    });

    lastNodeId = 'raw-field';
    yPos += nodeSpacing;
  }

  // =========================================================================
  // SIMILAR DOCUMENT LOOKUP
  // =========================================================================
  const similarDocStep = steps.find(s => s.name === 'similar_doc_lookup');
  if (similarDocStep) {
    const details = similarDocStep.details || {};

    nodes.push({
      id: 'similar-lookup',
      type: 'pipelineStage',
      position: { x: xCenter, y: yPos },
      data: {
        label: 'Similar Document Lookup',
        stageName: 'similar_doc_lookup',
        status: similarDocStep.status,
        duration: similarDocStep.duration_seconds,
        description: details.description || `Found ${details.similar_docs_found || 0} similar documents`,
        embeddingModel: details.embedding_model,
        details: {
          similar_docs_found: details.similar_docs_found,
          strategy_used: details.strategy_used,
          has_hints: details.has_hints,
        }
      }
    });

    edges.push({
      id: `${lastNodeId}-similar`,
      source: lastNodeId,
      target: 'similar-lookup',
      type: 'smoothstep',
    });

    lastNodeId = 'similar-lookup';
    yPos += nodeSpacing;
  }

  // =========================================================================
  // EXTRACTION + CLASSIFICATION (always shown as separate nodes)
  // =========================================================================
  const parallelStep = steps.find(s => s.name === 'parallel_extraction');
  if (parallelStep) {
    const details = parallelStep.details || {};

    // Build extraction description
    const extractionParts = [];
    if (details.test_results_count > 0) extractionParts.push(`${details.test_results_count} tests`);
    if (details.medications_count > 0) extractionParts.push(`${details.medications_count} meds`);
    if (details.findings_count > 0) extractionParts.push(`${details.findings_count} findings`);
    const extractionDesc = extractionParts.length > 0
      ? `Extracted ${extractionParts.join(', ')}` : 'LLM-based content extraction';

    // Build classification description
    const classType = details.document_type || 'unknown';
    const classConf = details.classification_confidence
      ? `${Math.round(details.classification_confidence * 100)}%`
      : '';
    const classDesc = classConf ? `${classType} (${classConf} confidence)` : classType;

    if (details.parallel) {
      // PARALLEL: two nodes side by side with fork-join edges
      const xOffset = 220;

      nodes.push({
        id: 'content-extraction',
        type: 'pipelineStage',
        position: { x: xCenter - xOffset, y: yPos },
        data: {
          label: 'Content Extraction',
          stageName: 'parallel_extraction',
          status: details.extraction_success ? 'completed' : 'failed',
          duration: parallelStep.duration_seconds,
          description: extractionDesc,
          llmModel: details.llm_model,
          details: {
            test_results_count: details.test_results_count,
            medications_count: details.medications_count,
            findings_count: details.findings_count,
            patient_extracted: details.patient_extracted,
            raw_fields_count: details.raw_fields_count,
          }
        }
      });

      nodes.push({
        id: 'doc-classification',
        type: 'pipelineStage',
        position: { x: xCenter + xOffset, y: yPos },
        data: {
          label: 'Document Classification',
          stageName: 'classification',
          status: details.classification_success ? 'completed' : 'failed',
          duration: parallelStep.duration_seconds,
          description: classDesc,
          llmModel: details.llm_model,
          details: {
            document_type: details.document_type,
            classification_confidence: details.classification_confidence,
          }
        }
      });

      // Fork: previous node → both
      edges.push({
        id: `${lastNodeId}-extraction`,
        source: lastNodeId,
        target: 'content-extraction',
        type: 'smoothstep',
      });
      edges.push({
        id: `${lastNodeId}-classification`,
        source: lastNodeId,
        target: 'doc-classification',
        type: 'smoothstep',
      });

      // We'll join to the next node below — set both as "last" via a join node
      lastNodeId = 'parallel-join';
      yPos += nodeSpacing;

    } else {
      // SEQUENTIAL: two stacked nodes
      nodes.push({
        id: 'content-extraction',
        type: 'pipelineStage',
        position: { x: xCenter, y: yPos },
        data: {
          label: 'Content Extraction',
          stageName: 'parallel_extraction',
          status: details.extraction_success !== false ? parallelStep.status : 'failed',
          duration: parallelStep.duration_seconds,
          description: extractionDesc,
          llmModel: details.llm_model,
          details: {
            test_results_count: details.test_results_count,
            medications_count: details.medications_count,
            findings_count: details.findings_count,
            patient_extracted: details.patient_extracted,
            raw_fields_count: details.raw_fields_count,
          }
        }
      });

      edges.push({
        id: `${lastNodeId}-extraction`,
        source: lastNodeId,
        target: 'content-extraction',
        type: 'smoothstep',
      });

      yPos += nodeSpacing;

      nodes.push({
        id: 'doc-classification',
        type: 'pipelineStage',
        position: { x: xCenter, y: yPos },
        data: {
          label: 'Document Classification',
          stageName: 'classification',
          status: details.classification_success !== false ? parallelStep.status : 'failed',
          description: classDesc,
          llmModel: details.llm_model,
          details: {
            document_type: details.document_type,
            classification_confidence: details.classification_confidence,
          }
        }
      });

      edges.push({
        id: 'extraction-classification',
        source: 'content-extraction',
        target: 'doc-classification',
        type: 'smoothstep',
      });

      lastNodeId = 'doc-classification';
      yPos += nodeSpacing;
    }
  }

  // =========================================================================
  // Helper: connect from parallel fork or single lastNodeId
  // =========================================================================
  const addEdgesToTarget = (targetId) => {
    if (lastNodeId === 'parallel-join') {
      // Join: both parallel nodes → target
      edges.push({
        id: `content-extraction-${targetId}`,
        source: 'content-extraction',
        target: targetId,
        type: 'smoothstep',
      });
      edges.push({
        id: `doc-classification-${targetId}`,
        source: 'doc-classification',
        target: targetId,
        type: 'smoothstep',
      });
    } else {
      edges.push({
        id: `${lastNodeId}-${targetId}`,
        source: lastNodeId,
        target: targetId,
        type: 'smoothstep',
      });
    }
  };

  // =========================================================================
  // ENRICHMENT (if applicable)
  // =========================================================================
  const enrichmentStep = steps.find(s => s.name.includes('enrichment'));
  if (enrichmentStep) {
    const details = enrichmentStep.details || {};

    nodes.push({
      id: 'enrichment',
      type: 'pipelineStage',
      position: { x: xCenter, y: yPos },
      data: {
        label: enrichmentStep.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        stageName: enrichmentStep.name,
        status: enrichmentStep.status,
        duration: enrichmentStep.duration_seconds,
        description: details.description || (enrichmentStep.status === 'skipped' ? details.reason : 'Type-specific enrichment'),
        llmModel: details.llm_model,
        details: details
      }
    });

    addEdgesToTarget('enrichment');

    lastNodeId = 'enrichment';
    yPos += nodeSpacing;
  }

  // =========================================================================
  // DATABASE VALIDATION
  // =========================================================================
  const dbValidationStep = steps.find(s => s.name === 'database_validation');
  if (dbValidationStep) {
    const details = dbValidationStep.details || {};
    const verified = details.verified_count || 0;
    const unverified = details.unverified_count || 0;
    const strengthMismatch = details.strength_mismatch_count || 0;
    const total = details.total_items_checked || 0;
    const dbName = details.database || 'Medical DB';

    let description = details.description || 'Validated against medical coding database';
    if (total > 0) {
      description = `${dbName}: ${verified}/${total} verified`;
      if (strengthMismatch > 0) {
        description += ` — ${strengthMismatch} strength mismatch`;
      }
      if (unverified > 0) {
        description += ` — ${unverified} unverified`;
      }
    }

    nodes.push({
      id: 'db-validation',
      type: 'pipelineStage',
      position: { x: xCenter, y: yPos },
      data: {
        label: `${dbName} Validation`,
        stageName: 'database_validation',
        status: dbValidationStep.status,
        duration: dbValidationStep.duration_seconds,
        description: description,
        warning: details.warning,
        details: {
          database: details.database,
          total_items_checked: total,
          verified_count: verified,
          unverified_count: unverified,
          strength_mismatch_count: strengthMismatch,
          rxnorm_codes_found: details.rxnorm_codes_found,
          loinc_codes_found: details.loinc_codes_found,
          ocr_corrections: details.ocr_corrections,
          interactions_detected: details.interactions_detected,
        }
      }
    });

    addEdgesToTarget('db-validation');

    lastNodeId = 'db-validation';
    yPos += nodeSpacing;
  }

  // =========================================================================
  // VECTOR STORE
  // =========================================================================
  const vectorStep = steps.find(s => s.name === 'vector_store');
  if (vectorStep) {
    const details = vectorStep.details || {};

    nodes.push({
      id: 'vector-store',
      type: 'pipelineStage',
      position: { x: xCenter, y: yPos },
      data: {
        label: 'Vector Store',
        stageName: 'vector_store',
        status: vectorStep.status,
        duration: vectorStep.duration_seconds,
        description: details.description || 'Stored extraction for future learning',
        embeddingModel: details.embedding_model,
        details: {
          doc_id: details.doc_id,
          confidence_stored: details.confidence_stored,
        }
      }
    });

    addEdgesToTarget('vector-store');

    lastNodeId = 'vector-store';
    yPos += nodeSpacing;
  }

  // =========================================================================
  // END NODE
  // =========================================================================
  yPos += 20;

  const endStats = [];
  if (result.extracted_values?.length > 0) {
    endStats.push({ value: result.extracted_values.length, label: 'values' });
  }
  if (result.raw_fields && Object.keys(result.raw_fields).length > 0) {
    endStats.push({ value: Object.keys(result.raw_fields).length, label: 'fields' });
  }
  if (result.total_time) {
    endStats.push({ value: `${result.total_time.toFixed(1)}s`, label: 'total' });
  }

  nodes.push({
    id: 'end',
    type: 'end',
    position: { x: xCenter, y: yPos },
    data: {
      label: job.status === 'completed' ? 'Processing Complete' :
             job.status === 'failed' ? 'Processing Failed' : 'Output',
      status: job.status,
      confidence: result.confidence,
      stats: endStats
    }
  });

  addEdgesToTarget('end');

  // =========================================================================
  // REVIEW NODE (if required)
  // =========================================================================
  if (result.requires_review) {
    nodes.push({
      id: 'review',
      type: 'review',
      position: { x: xCenter + 300, y: yPos },
      data: {
        label: 'Human Review Required',
        reasons: result.review_reasons || ['Low confidence']
      }
    });

    edges.push({
      id: 'end-review',
      source: 'end',
      target: 'review',
      type: 'smoothstep',
      style: { stroke: '#F59E0B', strokeDasharray: '5,5' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#F59E0B' },
    });
  }

  return { nodes, edges };
};

// ============================================================================
// Sidebar Component
// ============================================================================

const FlowSidebar = ({ job }) => {
  const result = job?.result || {};
  const steps = job?.workflow_steps || [];
  const textResult = result.text_result || {};

  const totalDuration = steps.reduce((sum, s) => sum + (s.duration_seconds || 0), 0);
  const completedSteps = steps.filter(s => s.status === 'completed').length;

  return (
    <div className="flow-sidebar">
      <h3>Pipeline Summary</h3>

      <div className="sidebar-section">
        <div className="summary-stats">
          <div className="stat">
            <span className="stat-value">{completedSteps}/{steps.length}</span>
            <span className="stat-label">Stages</span>
          </div>
          <div className="stat">
            <span className="stat-value">{totalDuration.toFixed(1)}s</span>
            <span className="stat-label">Total Time</span>
          </div>
          <div className="stat">
            <span className="stat-value">{Math.round((result.confidence || 0) * 100)}%</span>
            <span className="stat-label">Confidence</span>
          </div>
        </div>
      </div>

      {/* Extraction Method */}
      <div className="sidebar-section">
        <h4>Extraction Method</h4>
        <div className="method-badge-large">
          {steps.some(s => s.name === 'cloud_extraction') ? (
            <>
              <Cloud size={16} />
              <span>Azure GPT-4o Vision (Cloud)</span>
            </>
          ) : textResult.extraction_method?.includes('consensus') ? (
            <>
              <SplitSquareHorizontal size={16} />
              <span>Consensus (VLM + OCR)</span>
            </>
          ) : textResult.extraction_method?.includes('vlm') ? (
            <>
              <Eye size={16} />
              <span>Vision Language Model</span>
            </>
          ) : textResult.extraction_method ? (
            <>
              <Type size={16} />
              <span>OCR</span>
            </>
          ) : null}
        </div>
      </div>

      {/* Models Used */}
      <div className="sidebar-section">
        <h4>Models Used</h4>
        <div className="models-list">
          {steps.map((step, idx) => {
            const details = step.details || {};
            const models = [];
            if (details.vlm_model) models.push({ type: 'vlm', name: details.vlm_model });
            if (details.llm_model) models.push({ type: 'llm', name: details.llm_model });
            if (details.deployment) models.push({ type: 'llm', name: details.deployment });
            if (details.embedding_model) models.push({ type: 'embedding', name: details.embedding_model });

            if (models.length === 0) return null;

            return (
              <div key={idx} className="model-row">
                <span className="model-stage">{step.name.replace(/_/g, ' ')}</span>
                <div className="model-badges">
                  {models.map((m, i) => (
                    <span key={i} className={`model-badge-sm ${m.type}`}>{m.name}</span>
                  ))}
                </div>
              </div>
            );
          }).filter(Boolean)}
        </div>
      </div>

      {/* Stage Timeline */}
      <div className="sidebar-section">
        <h4>Stage Timeline</h4>
        <div className="stage-timeline">
          {steps.map((step, idx) => (
            <div key={idx} className={`timeline-item ${step.status}`}>
              <div className="timeline-dot" style={{ backgroundColor: getStageColor(step.name, step.status) }} />
              <div className="timeline-content">
                <span className="timeline-name">{step.name.replace(/_/g, ' ')}</span>
                <span className="timeline-duration">
                  {step.duration_seconds ?
                    (step.duration_seconds < 1 ? `${Math.round(step.duration_seconds * 1000)}ms` : `${step.duration_seconds.toFixed(1)}s`) :
                    '-'}
                </span>
              </div>
              <span className={`timeline-status ${step.status}`}>
                {step.status === 'completed' ? <CheckCircle size={12} /> :
                 step.status === 'failed' ? <AlertCircle size={12} /> :
                 step.status === 'skipped' ? <span>-</span> :
                 <Loader2 size={12} className="spinning" />}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Warnings */}
      {result.warnings?.length > 0 && (
        <div className="sidebar-section warnings-section">
          <h4>Warnings</h4>
          {result.warnings.map((warning, idx) => (
            <div key={idx} className="warning-item">
              <AlertTriangle size={12} />
              <span>{warning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

const ProcessFlow = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();

  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showSidebar, setShowSidebar] = useState(true);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const fetchJob = useCallback(async () => {
    if (!jobId) {
      setLoading(false);
      return;
    }

    try {
      const data = await getJobStatus(jobId);
      setJob(data);

      const { nodes: flowNodes, edges: flowEdges } = generateFlowFromJob(data);
      setNodes(flowNodes);
      setEdges(flowEdges);
    } catch (err) {
      console.error('Failed to fetch job:', err);
    } finally {
      setLoading(false);
    }
  }, [jobId, setNodes, setEdges]);

  useEffect(() => {
    fetchJob();

    const interval = setInterval(() => {
      if (job?.status === 'processing' || job?.status === 'pending') {
        fetchJob();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [fetchJob, job?.status]);

  const miniMapNodeColor = useCallback((node) => {
    switch (node.data?.status) {
      case 'completed': return '#10B981';
      case 'running': return '#3B82F6';
      case 'failed': return '#EF4444';
      case 'skipped': return '#9CA3AF';
      default: return '#6B7280';
    }
  }, []);

  if (loading) {
    return (
      <div className="flow-loading">
        <Loader2 size={32} className="spinning" />
        <p>Loading pipeline flow...</p>
      </div>
    );
  }

  return (
    <div className="process-flow-page">
      <div className="flow-header">
        <div className="header-left">
          <button className="btn btn-ghost" onClick={() => navigate(-1)}>
            <ArrowLeft size={20} />
            Back
          </button>
          <div className="header-info">
            <h1>Document Processing Pipeline</h1>
            {job && (
              <p className="header-subtitle">
                {job.file_name} • {job.document_type}
                {job.workflow_steps?.length > 0 && ` • ${job.workflow_steps.length} stages`}
              </p>
            )}
          </div>
        </div>
        <div className="header-actions">
          <button
            className={`btn btn-secondary ${showSidebar ? 'active' : ''}`}
            onClick={() => setShowSidebar(!showSidebar)}
          >
            <MessageSquare size={18} />
            {showSidebar ? 'Hide' : 'Show'} Details
          </button>
          <div className="flow-legend">
            <div className="legend-item">
              <span className="legend-dot completed"></span>
              <span>Completed</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot running"></span>
              <span>Running</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot skipped"></span>
              <span>Skipped</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot failed"></span>
              <span>Failed</span>
            </div>
          </div>
        </div>
      </div>

      <div className="flow-body">
        <div className={`flow-container ${showSidebar ? 'with-sidebar' : ''}`}>
          {!jobId ? (
            <div className="flow-empty">
              <GitBranch size={48} />
              <h3>No document selected</h3>
              <p>Select a document from the documents list to view its processing flow</p>
              <button className="btn btn-primary" onClick={() => navigate('/documents')}>
                View Documents
              </button>
            </div>
          ) : (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              nodeTypes={nodeTypes}
              fitView
              fitViewOptions={{ padding: 0.3 }}
              attributionPosition="bottom-left"
              defaultEdgeOptions={{
                type: 'smoothstep',
                style: { stroke: '#6B7280', strokeWidth: 2 },
                markerEnd: { type: MarkerType.ArrowClosed, color: '#6B7280' },
              }}
            >
              <Background color="#E5E7EB" gap={20} />
              <Controls />
              <MiniMap
                nodeColor={miniMapNodeColor}
                maskColor="rgba(0, 0, 0, 0.1)"
              />
            </ReactFlow>
          )}
        </div>

        {showSidebar && job && <FlowSidebar job={job} />}
      </div>
    </div>
  );
};

export default ProcessFlow;
