// ============================================================================
// Uploads Screen - List of recent uploads and their status
// ============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { listJobs, getJobStatus } from '../services/api';

// Status configurations
const STATUS_CONFIG = {
  pending: {
    icon: 'time-outline',
    color: '#fbbf24',
    label: 'Pending',
  },
  processing: {
    icon: 'sync-outline',
    color: '#0ea5e9',
    label: 'Processing',
  },
  completed: {
    icon: 'checkmark-circle',
    color: '#10b981',
    label: 'Completed',
  },
  failed: {
    icon: 'close-circle',
    color: '#ef4444',
    label: 'Failed',
  },
};

// Document type icons
const TYPE_ICONS = {
  lab: 'flask-outline',
  radiology: 'scan-outline',
  prescription: 'medical-outline',
  pathology: 'analytics-outline',
  auto: 'sparkles-outline',
  unknown: 'document-outline',
};

export default function UploadsScreen({ navigation, route }) {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // Fetch jobs
  const fetchJobs = async (showLoader = true) => {
    try {
      if (showLoader) setLoading(true);
      setError(null);

      const response = await listJobs();
      // Sort by created_at descending (newest first)
      const sortedJobs = (response.jobs || []).sort((a, b) => {
        return new Date(b.created_at) - new Date(a.created_at);
      });
      setJobs(sortedJobs);
    } catch (err) {
      console.error('Failed to fetch jobs:', err);
      setError('Failed to load uploads. Pull to retry.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Refresh on pull
  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchJobs(false);
  }, []);

  // Fetch on mount and when screen is focused
  useFocusEffect(
    useCallback(() => {
      fetchJobs(jobs.length === 0);

      // Poll for processing jobs
      const interval = setInterval(() => {
        const hasProcessingJobs = jobs.some(
          (job) => job.status === 'pending' || job.status === 'processing'
        );
        if (hasProcessingJobs) {
          fetchJobs(false);
        }
      }, 3000);

      return () => clearInterval(interval);
    }, [jobs.length])
  );

  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString();
  };

  // Render job item
  const renderJobItem = ({ item }) => {
    const status = STATUS_CONFIG[item.status] || STATUS_CONFIG.pending;
    const typeIcon = TYPE_ICONS[item.document_type] || TYPE_ICONS.unknown;
    const isProcessing = item.status === 'processing' || item.status === 'pending';

    return (
      <TouchableOpacity
        style={styles.jobItem}
        onPress={() => navigation.navigate('JobDetail', { jobId: item.job_id })}
        activeOpacity={0.7}
      >
        {/* Document Type Icon */}
        <View style={[styles.typeIcon, { backgroundColor: `${status.color}20` }]}>
          <Ionicons name={typeIcon} size={24} color={status.color} />
        </View>

        {/* Job Info */}
        <View style={styles.jobInfo}>
          <Text style={styles.fileName} numberOfLines={1}>
            {item.file_name || 'Document'}
          </Text>
          <View style={styles.jobMeta}>
            <Text style={styles.docType}>
              {(item.document_type || 'auto').charAt(0).toUpperCase() +
                (item.document_type || 'auto').slice(1)}
            </Text>
            <Text style={styles.separator}>•</Text>
            <Text style={styles.timestamp}>{formatDate(item.created_at)}</Text>
          </View>
        </View>

        {/* Status */}
        <View style={styles.statusContainer}>
          {isProcessing ? (
            <ActivityIndicator size="small" color={status.color} />
          ) : (
            <Ionicons name={status.icon} size={24} color={status.color} />
          )}
        </View>
      </TouchableOpacity>
    );
  };

  // Render empty state
  const renderEmpty = () => {
    if (loading) return null;

    return (
      <View style={styles.emptyContainer}>
        <Ionicons name="cloud-upload-outline" size={64} color="#4b5563" />
        <Text style={styles.emptyTitle}>No uploads yet</Text>
        <Text style={styles.emptyText}>
          Capture a document to get started
        </Text>
        <TouchableOpacity
          style={styles.emptyButton}
          onPress={() => navigation.navigate('Capture')}
        >
          <Ionicons name="camera-outline" size={20} color="#fff" />
          <Text style={styles.emptyButtonText}>Open Camera</Text>
        </TouchableOpacity>
      </View>
    );
  };

  // Render error state
  if (error && jobs.length === 0) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="cloud-offline-outline" size={64} color="#ef4444" />
        <Text style={styles.errorTitle}>Connection Error</Text>
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={() => fetchJobs()}>
          <Text style={styles.retryButtonText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {loading && jobs.length === 0 ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#0ea5e9" />
          <Text style={styles.loadingText}>Loading uploads...</Text>
        </View>
      ) : (
        <FlatList
          data={jobs}
          renderItem={renderJobItem}
          keyExtractor={(item) => item.job_id}
          contentContainerStyle={[
            styles.listContent,
            jobs.length === 0 && styles.listContentEmpty,
          ]}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor="#0ea5e9"
              colors={['#0ea5e9']}
            />
          }
          ListEmptyComponent={renderEmpty}
          ItemSeparatorComponent={() => <View style={styles.separator} />}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
  },
  listContent: {
    padding: 16,
  },
  listContentEmpty: {
    flex: 1,
  },
  jobItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#1f2937',
    borderRadius: 12,
  },
  typeIcon: {
    width: 48,
    height: 48,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  jobInfo: {
    flex: 1,
    marginLeft: 12,
  },
  fileName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 4,
  },
  jobMeta: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  docType: {
    fontSize: 13,
    color: '#9ca3af',
  },
  separator: {
    height: 12,
  },
  timestamp: {
    fontSize: 13,
    color: '#6b7280',
    marginLeft: 6,
  },
  statusContainer: {
    width: 32,
    alignItems: 'center',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#9ca3af',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#9ca3af',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 24,
  },
  emptyButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#0ea5e9',
    borderRadius: 12,
  },
  emptyButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
    backgroundColor: '#111827',
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
});
