// ============================================================================
// Status Badge Component
// ============================================================================

import React from 'react';
import { Clock, Loader2, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

const statusConfig = {
  pending: { icon: Clock, label: 'Pending', className: 'badge-status-pending' },
  processing: { icon: Loader2, label: 'Processing', className: 'badge-status-processing' },
  completed: { icon: CheckCircle, label: 'Completed', className: 'badge-status-completed' },
  failed: { icon: XCircle, label: 'Failed', className: 'badge-status-failed' },
  review_required: { icon: AlertTriangle, label: 'Review Required', className: 'badge-status-review' },
};

const StatusBadge = ({ status }) => {
  const config = statusConfig[status] || statusConfig.pending;
  const Icon = config.icon;
  const isSpinning = status === 'processing';

  return (
    <span className={`badge ${config.className}`}>
      <Icon size={14} className={isSpinning ? 'spinning' : ''} />
      {config.label}
    </span>
  );
};

export default StatusBadge;
