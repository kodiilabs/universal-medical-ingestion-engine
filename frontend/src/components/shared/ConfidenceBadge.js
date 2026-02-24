// ============================================================================
// Confidence Badge Component
// ============================================================================

import React from 'react';

const getConfidenceLevel = (score) => {
  if (score >= 0.85) return { level: 'high', label: 'High' };
  if (score >= 0.70) return { level: 'medium', label: 'Medium' };
  if (score >= 0.50) return { level: 'low', label: 'Low' };
  return { level: 'critical', label: 'Critical' };
};

const ConfidenceBadge = ({ score, showLabel = true, size = 'md' }) => {
  const { level, label } = getConfidenceLevel(score);
  const percentage = Math.round(score * 100);

  return (
    <span className={`badge badge-confidence-${level} ${size === 'sm' ? 'text-xs' : ''}`}>
      {percentage}%
      {showLabel && <span style={{ marginLeft: 4, opacity: 0.8 }}>{label}</span>}
    </span>
  );
};

export default ConfidenceBadge;
