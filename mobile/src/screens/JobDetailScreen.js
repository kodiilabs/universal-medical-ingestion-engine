// ============================================================================
// Job Detail Screen - Show processing details and results
// ============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  RefreshControl,
  TouchableOpacity,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { getJobStatus, deleteJob } from '../services/api';

// Status configurations
const STATUS_CONFIG = {
  pending: { icon: 'time-outline', color: '#fbbf24', label: 'Pending' },
  processing: { icon: 'sync-outline', color: '#0ea5e9', label: 'Processing' },
  completed: { icon: 'checkmark-circle', color: '#10b981', label: 'Completed' },
  failed: { icon: 'close-circle', color: '#ef4444', label: 'Failed' },
  running: { icon: 'play-circle', color: '#0ea5e9', label: 'Running' },
};

export default function JobDetailScreen({ route, navigation }) {
  const { jobId } = route.params;
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // Fetch job details
  const fetchJob = async (showLoader = true) => {
    try {
      if (showLoader) setLoading(true);
      setError(null);

      const data = await getJobStatus(jobId);
      setJob(data);
    } catch (err) {
      console.error('Failed to fetch job:', err);
      setError('Failed to load job details');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Refresh on pull
  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchJob(false);
  }, []);

  // Fetch on mount and poll for processing jobs
  useFocusEffect(
    useCallback(() => {
      fetchJob();

      const interval = setInterval(() => {
        if (job?.status === 'pending' || job?.status === 'processing') {
          fetchJob(false);
        }
      }, 2000);

      return () => clearInterval(interval);
    }, [job?.status])
  );

  // Handle delete
  const handleDelete = async () => {
    try {
      await deleteJob(jobId);
      navigation.goBack();
    } catch (err) {
      console.error('Failed to delete job:', err);
    }
  };

  // Format confidence as percentage
  const formatConfidence = (confidence) => {
    if (typeof confidence !== 'number') return 'N/A';
    return `${Math.round(confidence * 100)}%`;
  };

  // Render workflow step
  const renderWorkflowStep = (step, index) => {
    const status = STATUS_CONFIG[step.status] || STATUS_CONFIG.pending;
    const isLast = index === (job?.workflow_steps?.length || 0) - 1;

    return (
      <View key={step.id} style={styles.stepContainer}>
        {/* Connector line */}
        {!isLast && <View style={[styles.stepLine, { backgroundColor: status.color }]} />}

        {/* Step indicator */}
        <View style={[styles.stepIndicator, { backgroundColor: `${status.color}30` }]}>
          {step.status === 'running' ? (
            <ActivityIndicator size="small" color={status.color} />
          ) : (
            <Ionicons name={status.icon} size={16} color={status.color} />
          )}
        </View>

        {/* Step content */}
        <View style={styles.stepContent}>
          <Text style={styles.stepName}>{step.name}</Text>
          {step.details && (
            <View style={styles.stepDetails}>
              {Object.entries(step.details).slice(0, 3).map(([key, value]) => (
                <Text key={key} style={styles.stepDetail}>
                  {key}: {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </Text>
              ))}
            </View>
          )}
        </View>
      </View>
    );
  };

  // Render extracted value
  const renderExtractedValue = (value, index) => {
    const confidence = value.confidence || 0;
    const isLow = confidence < 0.7;
    const isHigh = confidence >= 0.9;

    return (
      <View key={index} style={styles.valueItem}>
        <View style={styles.valueHeader}>
          <Text style={styles.valueName}>{value.field_name || 'Unknown'}</Text>
          <View
            style={[
              styles.confidenceBadge,
              isLow && styles.confidenceLow,
              isHigh && styles.confidenceHigh,
            ]}
          >
            <Text
              style={[
                styles.confidenceText,
                isLow && styles.confidenceTextLow,
                isHigh && styles.confidenceTextHigh,
              ]}
            >
              {formatConfidence(confidence)}
            </Text>
          </View>
        </View>
        <Text style={styles.valueText}>
          {value.value}
          {value.unit ? ` ${value.unit}` : ''}
        </Text>
        {value.abnormal_flag && (
          <View style={styles.abnormalBadge}>
            <Ionicons name="warning-outline" size={12} color="#ef4444" />
            <Text style={styles.abnormalText}>{value.abnormal_flag}</Text>
          </View>
        )}
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0ea5e9" />
        <Text style={styles.loadingText}>Loading job details...</Text>
      </View>
    );
  }

  if (error || !job) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="alert-circle-outline" size={64} color="#ef4444" />
        <Text style={styles.errorTitle}>Error</Text>
        <Text style={styles.errorText}>{error || 'Job not found'}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={() => fetchJob()}>
          <Text style={styles.retryButtonText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const status = STATUS_CONFIG[job.status] || STATUS_CONFIG.pending;
  const isProcessing = job.status === 'pending' || job.status === 'processing';
  const extractedValues = job.result?.extracted_values || [];

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor="#0ea5e9"
        />
      }
    >
      {/* Status Header */}
      <View style={[styles.statusCard, { borderColor: status.color }]}>
        <View style={styles.statusHeader}>
          {isProcessing ? (
            <ActivityIndicator size="large" color={status.color} />
          ) : (
            <Ionicons name={status.icon} size={48} color={status.color} />
          )}
          <View style={styles.statusInfo}>
            <Text style={[styles.statusLabel, { color: status.color }]}>
              {status.label}
            </Text>
            <Text style={styles.fileName}>{job.file_name}</Text>
            <Text style={styles.docType}>
              {(job.document_type || 'Auto').charAt(0).toUpperCase() +
                (job.document_type || 'Auto').slice(1)}
            </Text>
          </View>
        </View>

        {job.result?.confidence !== undefined && (
          <View style={styles.confidenceRow}>
            <Text style={styles.confidenceLabel}>Overall Confidence</Text>
            <Text style={styles.confidenceValue}>
              {formatConfidence(job.result.confidence)}
            </Text>
          </View>
        )}
      </View>

      {/* Workflow Steps */}
      {job.workflow_steps?.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Processing Steps</Text>
          <View style={styles.stepsContainer}>
            {job.workflow_steps.map(renderWorkflowStep)}
          </View>
        </View>
      )}

      {/* Extracted Values */}
      {extractedValues.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            Extracted Values ({extractedValues.length})
          </Text>
          <View style={styles.valuesContainer}>
            {extractedValues.slice(0, 10).map(renderExtractedValue)}
            {extractedValues.length > 10 && (
              <Text style={styles.moreText}>
                +{extractedValues.length - 10} more values
              </Text>
            )}
          </View>
        </View>
      )}

      {/* Clinical Summary */}
      {job.result?.clinical_summary && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Clinical Summary</Text>
          <View style={styles.summaryCard}>
            <Text style={styles.summaryText}>{job.result.clinical_summary}</Text>
          </View>
        </View>
      )}

      {/* Critical Findings */}
      {job.result?.critical_findings?.length > 0 && (
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: '#ef4444' }]}>
            Critical Findings
          </Text>
          <View style={styles.criticalCard}>
            {job.result.critical_findings.map((finding, index) => (
              <View key={index} style={styles.criticalItem}>
                <Ionicons name="alert-circle" size={16} color="#ef4444" />
                <Text style={styles.criticalText}>{finding}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Error Message */}
      {job.error && (
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: '#ef4444' }]}>Error</Text>
          <View style={styles.errorCard}>
            <Text style={styles.errorMessage}>{job.error}</Text>
          </View>
        </View>
      )}

      {/* Delete Button */}
      <TouchableOpacity style={styles.deleteButton} onPress={handleDelete}>
        <Ionicons name="trash-outline" size={20} color="#ef4444" />
        <Text style={styles.deleteButtonText}>Delete Job</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
  },
  content: {
    padding: 16,
    paddingBottom: 32,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#111827',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#9ca3af',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#111827',
    padding: 32,
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#ef4444',
    marginTop: 16,
  },
  errorText: {
    fontSize: 14,
    color: '#9ca3af',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 24,
  },
  retryButton: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#374151',
    borderRadius: 12,
  },
  retryButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  statusCard: {
    backgroundColor: '#1f2937',
    borderRadius: 16,
    padding: 20,
    borderWidth: 2,
    marginBottom: 16,
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusInfo: {
    marginLeft: 16,
    flex: 1,
  },
  statusLabel: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  fileName: {
    fontSize: 14,
    color: '#fff',
    marginBottom: 2,
  },
  docType: {
    fontSize: 13,
    color: '#9ca3af',
  },
  confidenceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#374151',
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#9ca3af',
  },
  confidenceValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#10b981',
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  stepsContainer: {
    backgroundColor: '#1f2937',
    borderRadius: 12,
    padding: 16,
  },
  stepContainer: {
    flexDirection: 'row',
    position: 'relative',
  },
  stepLine: {
    position: 'absolute',
    left: 12,
    top: 28,
    bottom: -12,
    width: 2,
  },
  stepIndicator: {
    width: 28,
    height: 28,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
  },
  stepContent: {
    flex: 1,
    marginLeft: 12,
    paddingBottom: 16,
  },
  stepName: {
    fontSize: 14,
    fontWeight: '500',
    color: '#fff',
  },
  stepDetails: {
    marginTop: 4,
  },
  stepDetail: {
    fontSize: 12,
    color: '#6b7280',
  },
  valuesContainer: {
    backgroundColor: '#1f2937',
    borderRadius: 12,
    padding: 12,
  },
  valueItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#374151',
  },
  valueHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  valueName: {
    fontSize: 13,
    color: '#9ca3af',
    flex: 1,
  },
  confidenceBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
    backgroundColor: '#374151',
  },
  confidenceLow: {
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
  },
  confidenceHigh: {
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
  },
  confidenceText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#9ca3af',
  },
  confidenceTextLow: {
    color: '#ef4444',
  },
  confidenceTextHigh: {
    color: '#10b981',
  },
  valueText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  abnormalBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 4,
  },
  abnormalText: {
    fontSize: 12,
    color: '#ef4444',
  },
  moreText: {
    fontSize: 13,
    color: '#6b7280',
    textAlign: 'center',
    marginTop: 12,
  },
  summaryCard: {
    backgroundColor: '#1f2937',
    borderRadius: 12,
    padding: 16,
  },
  summaryText: {
    fontSize: 14,
    color: '#d1d5db',
    lineHeight: 22,
  },
  criticalCard: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.3)',
  },
  criticalItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginBottom: 8,
  },
  criticalText: {
    fontSize: 14,
    color: '#fca5a5',
    flex: 1,
  },
  errorCard: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.3)',
  },
  errorMessage: {
    fontSize: 14,
    color: '#fca5a5',
  },
  deleteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    marginTop: 16,
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.3)',
  },
  deleteButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#ef4444',
  },
});
