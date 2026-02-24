import React, { useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './WorkflowVisualization.css';

// Custom node component for workflow steps
const WorkflowNode = ({ data }) => {
  const statusColors = {
    pending: '#6a6a8e',
    running: '#4a4aff',
    completed: '#4ade80',
    failed: '#f87171',
  };

  const statusIcons = {
    pending: '○',
    running: '◉',
    completed: '✓',
    failed: '✗',
  };

  return (
    <div className={`workflow-node ${data.status}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-icon" style={{ backgroundColor: statusColors[data.status] }}>
        {statusIcons[data.status]}
      </div>
      <div className="node-content">
        <div className="node-title">{data.label}</div>
        <div className="node-status">{data.status}</div>
        {data.details && (
          <div className="node-details">
            {Object.entries(data.details).slice(0, 3).map(([key, value]) => (
              <div key={key} className="detail-item">
                <span className="detail-key">{key}:</span>
                <span className="detail-value">{String(value).slice(0, 50)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
};

// Custom node for data display
const DataNode = ({ data }) => {
  return (
    <div className={`data-node ${data.type}`}>
      <Handle type="target" position={Position.Left} />
      <div className="data-header">
        <span className="data-icon">{data.icon}</span>
        <span className="data-title">{data.label}</span>
      </div>
      <div className="data-content">
        {data.items && data.items.slice(0, 5).map((item, i) => (
          <div key={i} className="data-item">{item}</div>
        ))}
        {data.items && data.items.length > 5 && (
          <div className="data-more">+{data.items.length - 5} more</div>
        )}
        {data.value && (
          <div className="data-value">{data.value}</div>
        )}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
};

const nodeTypes = {
  workflow: WorkflowNode,
  data: DataNode,
};

function WorkflowVisualization({ steps, extractedData, documentType }) {
  const { nodes, edges } = useMemo(() => {
    const nodes = [];
    const edges = [];

    // Define the workflow based on document type
    const workflowSteps = [
      { id: 'upload', label: 'Document Upload', status: 'completed' },
      { id: 'extract', label: 'Text Extraction', status: steps.find(s => s.name === 'Text Extraction')?.status || 'pending' },
      { id: 'process', label: `${documentType?.charAt(0).toUpperCase()}${documentType?.slice(1)} Processing`, status: steps.find(s => s.name.includes('Processing'))?.status || 'pending' },
      { id: 'structure', label: 'Structure Extraction', status: steps.find(s => s.name === 'Structure Extraction')?.status || 'pending' },
      { id: 'store', label: 'Vector Store', status: steps.find(s => s.name === 'Vector Store')?.status || 'pending' },
    ];

    // Add workflow nodes
    workflowSteps.forEach((step, index) => {
      nodes.push({
        id: step.id,
        type: 'workflow',
        position: { x: 100 + index * 220, y: 100 },
        data: {
          label: step.label,
          status: step.status,
        },
      });

      if (index > 0) {
        edges.push({
          id: `e-${workflowSteps[index - 1].id}-${step.id}`,
          source: workflowSteps[index - 1].id,
          target: step.id,
          animated: step.status === 'running',
          style: { stroke: step.status === 'completed' ? '#4ade80' : '#3a3a5e' },
        });
      }
    });

    // Add data output nodes if we have extracted data
    if (extractedData) {
      const dataOutputs = [];

      // Add summary node
      if (extractedData.clinical_summary) {
        dataOutputs.push({
          id: 'summary',
          label: 'Clinical Summary',
          icon: '📋',
          type: 'summary',
          value: extractedData.clinical_summary.slice(0, 100) + '...',
        });
      }

      // Add sections
      if (extractedData.sections) {
        Object.entries(extractedData.sections).forEach(([name, content]) => {
          if (content && (typeof content === 'string' ? content.length > 0 : Object.keys(content).length > 0)) {
            const items = Array.isArray(content)
              ? content.map(c => typeof c === 'object' ? Object.values(c)[0] : c)
              : typeof content === 'object'
                ? Object.entries(content).map(([k, v]) => `${k}: ${v}`)
                : [content];

            dataOutputs.push({
              id: `section-${name}`,
              label: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
              icon: name.includes('critical') ? '⚠️' : '📄',
              type: name.includes('critical') ? 'critical' : 'section',
              items: items.filter(Boolean).slice(0, 5),
            });
          }
        });
      }

      // Add confidence node
      if (extractedData.confidence) {
        dataOutputs.push({
          id: 'confidence',
          label: 'Confidence Score',
          icon: '📊',
          type: 'metric',
          value: `${Math.round(extractedData.confidence * 100)}%`,
        });
      }

      // Position data nodes below workflow
      dataOutputs.forEach((output, index) => {
        const col = index % 3;
        const row = Math.floor(index / 3);
        nodes.push({
          id: output.id,
          type: 'data',
          position: { x: 150 + col * 280, y: 280 + row * 180 },
          data: output,
        });

        // Connect from structure extraction to data nodes
        edges.push({
          id: `e-structure-${output.id}`,
          source: 'structure',
          target: output.id,
          style: { stroke: '#4a4aff', strokeDasharray: '5,5' },
        });
      });
    }

    return { nodes, edges };
  }, [steps, extractedData, documentType]);

  return (
    <div className="workflow-visualization">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        attributionPosition="bottom-left"
      >
        <Background color="#2a2a4e" gap={20} />
        <Controls />
      </ReactFlow>
    </div>
  );
}

export default WorkflowVisualization;
