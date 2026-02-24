// ============================================================================
// Preview Screen - Review, select type, and upload document
// ============================================================================

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { uploadAndAnalyze, startProcessing } from '../services/api';

const DOCUMENT_TYPES = [
  { value: 'auto', label: 'Auto-detect', icon: 'sparkles-outline' },
  { value: 'lab', label: 'Lab Report', icon: 'flask-outline' },
  { value: 'radiology', label: 'Radiology', icon: 'scan-outline' },
  { value: 'prescription', label: 'Prescription', icon: 'medical-outline' },
];

// Upload states
const STATES = {
  IDLE: 'idle',
  UPLOADING: 'uploading',
  ANALYZING: 'analyzing',
  PROCESSING: 'processing',
  SUCCESS: 'success',
  ERROR: 'error',
  QUALITY_WARNING: 'quality_warning',
};

export default function PreviewScreen({ route, navigation }) {
  const { imageUri } = route.params;
  const [documentType, setDocumentType] = useState('auto');
  const [uploadState, setUploadState] = useState(STATES.IDLE);
  const [statusMessage, setStatusMessage] = useState('');
  const [qualityData, setQualityData] = useState(null);
  const [jobId, setJobId] = useState(null);

  // Handle upload and processing
  const handleUpload = async (forceUpload = false) => {
    try {
      // Step 1: Upload and analyze
      setUploadState(STATES.UPLOADING);
      setStatusMessage('Uploading document...');

      const uploadResult = await uploadAndAnalyze(imageUri, forceUpload);

      // Check for quality warnings
      if (uploadResult.recommendation === 'warning' && !forceUpload) {
        setUploadState(STATES.QUALITY_WARNING);
        setQualityData(uploadResult);
        setStatusMessage(uploadResult.message);
        return;
      }

      // Step 2: Start processing
      setUploadState(STATES.PROCESSING);
      setStatusMessage('Processing document...');

      const processResult = await startProcessing(uploadResult.file_id, documentType);
      setJobId(processResult.job_id);

      // Step 3: Success
      setUploadState(STATES.SUCCESS);
      setStatusMessage('Document uploaded successfully!');

      // Navigate to uploads list after a short delay
      setTimeout(() => {
        navigation.navigate('Uploads', {
          screen: 'UploadsList',
          params: { refreshTrigger: Date.now() },
        });
      }, 1500);

    } catch (error) {
      console.error('Upload error:', error);

      // Handle quality rejection
      if (error.response?.data?.detail?.error === 'image_quality_too_low') {
        setUploadState(STATES.QUALITY_WARNING);
        setQualityData(error.response.data.detail);
        setStatusMessage(error.response.data.detail.message);
        return;
      }

      // Handle other errors
      setUploadState(STATES.ERROR);
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      setStatusMessage(typeof errorMessage === 'string' ? errorMessage : 'Upload failed. Please try again.');
    }
  };

  // Handle retake
  const handleRetake = () => {
    navigation.goBack();
  };

  // Render quality warning
  const renderQualityWarning = () => {
    if (uploadState !== STATES.QUALITY_WARNING) return null;

    const tips = qualityData?.tips || [];

    return (
      <View style={styles.warningContainer}>
        <View style={styles.warningHeader}>
          <Ionicons name="warning-outline" size={24} color="#d97706" />
          <Text style={styles.warningTitle}>Image Quality Issue</Text>
        </View>
        <Text style={styles.warningMessage}>{statusMessage}</Text>

        {tips.length > 0 && (
          <View style={styles.tipsList}>
            {tips.map((tip, index) => (
              <View key={index} style={styles.tipItem}>
                <Ionicons name="checkmark-circle-outline" size={16} color="#6b7280" />
                <Text style={styles.tipText}>{tip}</Text>
              </View>
            ))}
          </View>
        )}

        <View style={styles.warningActions}>
          <TouchableOpacity
            style={styles.retakeButton}
            onPress={handleRetake}
          >
            <Ionicons name="camera-outline" size={20} color="#0ea5e9" />
            <Text style={styles.retakeButtonText}>Retake Photo</Text>
          </TouchableOpacity>

          {qualityData?.can_force && (
            <TouchableOpacity
              style={styles.forceButton}
              onPress={() => handleUpload(true)}
            >
              <Text style={styles.forceButtonText}>Try Anyway</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    );
  };

  // Render upload state overlay
  const renderStateOverlay = () => {
    if (uploadState === STATES.IDLE || uploadState === STATES.QUALITY_WARNING) return null;

    let icon = 'cloud-upload-outline';
    let iconColor = '#0ea5e9';
    let showSpinner = true;

    if (uploadState === STATES.SUCCESS) {
      icon = 'checkmark-circle';
      iconColor = '#10b981';
      showSpinner = false;
    } else if (uploadState === STATES.ERROR) {
      icon = 'close-circle';
      iconColor = '#ef4444';
      showSpinner = false;
    }

    return (
      <View style={styles.stateOverlay}>
        <View style={styles.stateCard}>
          {showSpinner ? (
            <ActivityIndicator size="large" color="#0ea5e9" />
          ) : (
            <Ionicons name={icon} size={48} color={iconColor} />
          )}
          <Text style={styles.stateMessage}>{statusMessage}</Text>

          {uploadState === STATES.ERROR && (
            <View style={styles.errorActions}>
              <TouchableOpacity
                style={styles.retryButton}
                onPress={() => handleUpload()}
              >
                <Text style={styles.retryButtonText}>Retry</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={handleRetake}
              >
                <Text style={styles.cancelButtonText}>Retake</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      </View>
    );
  };

  const isUploading = [STATES.UPLOADING, STATES.ANALYZING, STATES.PROCESSING].includes(uploadState);

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {/* Image Preview */}
        <View style={styles.imageContainer}>
          <Image source={{ uri: imageUri }} style={styles.image} resizeMode="contain" />
        </View>

        {/* Quality Warning */}
        {renderQualityWarning()}

        {/* Document Type Selector */}
        {uploadState !== STATES.QUALITY_WARNING && (
          <View style={styles.typeSelector}>
            <Text style={styles.sectionTitle}>Document Type</Text>
            <View style={styles.typeGrid}>
              {DOCUMENT_TYPES.map((type) => (
                <TouchableOpacity
                  key={type.value}
                  style={[
                    styles.typeButton,
                    documentType === type.value && styles.typeButtonActive,
                  ]}
                  onPress={() => setDocumentType(type.value)}
                  disabled={isUploading}
                >
                  <Ionicons
                    name={type.icon}
                    size={24}
                    color={documentType === type.value ? '#0ea5e9' : '#6b7280'}
                  />
                  <Text
                    style={[
                      styles.typeButtonText,
                      documentType === type.value && styles.typeButtonTextActive,
                    ]}
                  >
                    {type.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}
      </ScrollView>

      {/* Bottom Actions */}
      {uploadState !== STATES.QUALITY_WARNING && uploadState !== STATES.SUCCESS && (
        <View style={styles.bottomActions}>
          <TouchableOpacity
            style={styles.retakeAction}
            onPress={handleRetake}
            disabled={isUploading}
          >
            <Ionicons name="refresh-outline" size={20} color="#6b7280" />
            <Text style={styles.retakeActionText}>Retake</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.uploadAction, isUploading && styles.uploadActionDisabled]}
            onPress={() => handleUpload()}
            disabled={isUploading}
          >
            {isUploading ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Ionicons name="cloud-upload-outline" size={20} color="#fff" />
            )}
            <Text style={styles.uploadActionText}>
              {isUploading ? 'Uploading...' : 'Upload & Process'}
            </Text>
          </TouchableOpacity>
        </View>
      )}

      {/* State Overlay */}
      {renderStateOverlay()}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 100,
  },
  imageContainer: {
    backgroundColor: '#000',
    aspectRatio: 3 / 4,
    maxHeight: 400,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  typeSelector: {
    padding: 16,
  },
  typeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  typeButton: {
    flex: 1,
    minWidth: '45%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 14,
    backgroundColor: '#1f2937',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  typeButtonActive: {
    borderColor: '#0ea5e9',
    backgroundColor: 'rgba(14, 165, 233, 0.1)',
  },
  typeButtonText: {
    fontSize: 14,
    color: '#9ca3af',
    fontWeight: '500',
  },
  typeButtonTextActive: {
    color: '#0ea5e9',
  },
  bottomActions: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    gap: 12,
    padding: 16,
    backgroundColor: '#111827',
    borderTopWidth: 1,
    borderTopColor: '#374151',
  },
  retakeAction: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    paddingHorizontal: 20,
    backgroundColor: '#1f2937',
    borderRadius: 12,
  },
  retakeActionText: {
    fontSize: 16,
    color: '#9ca3af',
    fontWeight: '500',
  },
  uploadAction: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    backgroundColor: '#0ea5e9',
    borderRadius: 12,
  },
  uploadActionDisabled: {
    opacity: 0.7,
  },
  uploadActionText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '600',
  },
  stateOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  stateCard: {
    backgroundColor: '#1f2937',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    width: '100%',
    maxWidth: 300,
  },
  stateMessage: {
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
    marginTop: 16,
  },
  errorActions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 24,
  },
  retryButton: {
    paddingVertical: 10,
    paddingHorizontal: 24,
    backgroundColor: '#0ea5e9',
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  cancelButton: {
    paddingVertical: 10,
    paddingHorizontal: 24,
    backgroundColor: '#374151',
    borderRadius: 8,
  },
  cancelButtonText: {
    color: '#9ca3af',
    fontWeight: '500',
  },
  warningContainer: {
    margin: 16,
    padding: 16,
    backgroundColor: 'rgba(217, 119, 6, 0.1)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(217, 119, 6, 0.3)',
  },
  warningHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  warningTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#d97706',
  },
  warningMessage: {
    fontSize: 14,
    color: '#fbbf24',
    marginBottom: 12,
  },
  tipsList: {
    marginBottom: 16,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 6,
  },
  tipText: {
    fontSize: 13,
    color: '#9ca3af',
    flex: 1,
  },
  warningActions: {
    flexDirection: 'row',
    gap: 12,
  },
  retakeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    backgroundColor: '#1f2937',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#0ea5e9',
  },
  retakeButtonText: {
    color: '#0ea5e9',
    fontWeight: '600',
  },
  forceButton: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: 'transparent',
    borderRadius: 8,
  },
  forceButtonText: {
    color: '#9ca3af',
    fontWeight: '500',
  },
});
