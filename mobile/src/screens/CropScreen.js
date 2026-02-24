// ============================================================================
// Crop Screen - Trim dead space from captured documents
// ============================================================================

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  Dimensions,
  PanResponder,
  Alert,
} from 'react-native';
import * as ImageManipulator from 'expo-image-manipulator';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const HANDLE_SIZE = 28;
const HANDLE_HIT = 44; // larger touch target
const MIN_CROP = 60; // minimum crop box size in screen px
const TOP_BAR_HEIGHT = 56;
const BOTTOM_BAR_HEIGHT = 80;

// Available area for the image
const IMAGE_AREA_HEIGHT = SCREEN_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT - 100;

export default function CropScreen({ route, navigation }) {
  const { imageUri } = route.params;

  // Image layout within the screen
  const [imageLayout, setImageLayout] = useState(null);
  const [imageSize, setImageSize] = useState(null);

  // Crop box in screen coordinates (relative to image container)
  const [cropBox, setCropBox] = useState(null);

  // Refs that the PanResponder can read (always current)
  const cropBoxRef = useRef(null);
  const imageLayoutRef = useRef(null);
  const activeHandle = useRef(null);
  const startCrop = useRef(null);

  // Keep refs in sync with state
  useEffect(() => {
    cropBoxRef.current = cropBox;
  }, [cropBox]);
  useEffect(() => {
    imageLayoutRef.current = imageLayout;
  }, [imageLayout]);

  // Load image dimensions
  useEffect(() => {
    Image.getSize(
      imageUri,
      (width, height) => {
        setImageSize({ width, height });

        // Fit image to available screen area
        const maxW = SCREEN_WIDTH - 32;
        const maxH = IMAGE_AREA_HEIGHT;
        const scale = Math.min(maxW / width, maxH / height);
        const displayW = width * scale;
        const displayH = height * scale;
        const offsetX = (SCREEN_WIDTH - displayW) / 2;
        const offsetY = (IMAGE_AREA_HEIGHT - displayH) / 2;

        const layout = { width: displayW, height: displayH, offsetX, offsetY, scale };
        setImageLayout(layout);
        imageLayoutRef.current = layout;

        // Initialize crop box with 8% inset
        const insetX = displayW * 0.08;
        const insetY = displayH * 0.08;
        const initialCrop = {
          x: insetX,
          y: insetY,
          width: displayW - insetX * 2,
          height: displayH - insetY * 2,
        };
        setCropBox(initialCrop);
        cropBoxRef.current = initialCrop;
      },
      (error) => {
        console.error('Failed to get image size:', error);
        navigation.replace('Preview', { imageUri });
      }
    );
  }, [imageUri]);

  // Determine which handle a touch is near (reads from ref)
  const getHandleFromRef = (touchX, touchY) => {
    const box = cropBoxRef.current;
    if (!box) return null;

    const corners = {
      topLeft: { x: box.x, y: box.y },
      topRight: { x: box.x + box.width, y: box.y },
      bottomLeft: { x: box.x, y: box.y + box.height },
      bottomRight: { x: box.x + box.width, y: box.y + box.height },
    };

    for (const [name, pos] of Object.entries(corners)) {
      if (
        Math.abs(touchX - pos.x) < HANDLE_HIT &&
        Math.abs(touchY - pos.y) < HANDLE_HIT
      ) {
        return name;
      }
    }

    // Check edges
    const { x, y, width, height } = box;
    if (touchY > y && touchY < y + height) {
      if (Math.abs(touchX - x) < HANDLE_HIT) return 'left';
      if (Math.abs(touchX - (x + width)) < HANDLE_HIT) return 'right';
    }
    if (touchX > x && touchX < x + width) {
      if (Math.abs(touchY - y) < HANDLE_HIT) return 'top';
      if (Math.abs(touchY - (y + height)) < HANDLE_HIT) return 'bottom';
    }

    // Touch inside box = move entire box
    if (
      touchX > x + HANDLE_HIT &&
      touchX < x + width - HANDLE_HIT &&
      touchY > y + HANDLE_HIT &&
      touchY < y + height - HANDLE_HIT
    ) {
      return 'move';
    }

    return null;
  };

  // Clamp crop box to image bounds (reads from ref)
  const clampBoxFromRef = (box) => {
    const layout = imageLayoutRef.current;
    if (!layout) return box;
    const { width: imgW, height: imgH } = layout;

    let { x, y, width, height } = box;

    width = Math.max(MIN_CROP, width);
    height = Math.max(MIN_CROP, height);
    x = Math.max(0, Math.min(x, imgW - width));
    y = Math.max(0, Math.min(y, imgH - height));
    width = Math.min(width, imgW - x);
    height = Math.min(height, imgH - y);

    return { x, y, width, height };
  };

  // PanResponder — created once, reads current state via refs
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,

      onPanResponderGrant: (evt) => {
        if (!cropBoxRef.current || !imageLayoutRef.current) return;
        const touch = evt.nativeEvent;
        activeHandle.current = getHandleFromRef(touch.locationX, touch.locationY);
        startCrop.current = { ...cropBoxRef.current };
      },

      onPanResponderMove: (evt, gesture) => {
        if (!activeHandle.current || !startCrop.current || !imageLayoutRef.current) return;

        const { dx, dy } = gesture;
        const s = startCrop.current;
        let newBox = { ...s };

        switch (activeHandle.current) {
          case 'topLeft':
            newBox.x = s.x + dx;
            newBox.y = s.y + dy;
            newBox.width = s.width - dx;
            newBox.height = s.height - dy;
            break;
          case 'topRight':
            newBox.y = s.y + dy;
            newBox.width = s.width + dx;
            newBox.height = s.height - dy;
            break;
          case 'bottomLeft':
            newBox.x = s.x + dx;
            newBox.width = s.width - dx;
            newBox.height = s.height + dy;
            break;
          case 'bottomRight':
            newBox.width = s.width + dx;
            newBox.height = s.height + dy;
            break;
          case 'left':
            newBox.x = s.x + dx;
            newBox.width = s.width - dx;
            break;
          case 'right':
            newBox.width = s.width + dx;
            break;
          case 'top':
            newBox.y = s.y + dy;
            newBox.height = s.height - dy;
            break;
          case 'bottom':
            newBox.height = s.height + dy;
            break;
          case 'move':
            newBox.x = s.x + dx;
            newBox.y = s.y + dy;
            break;
        }

        const clamped = clampBoxFromRef(newBox);
        cropBoxRef.current = clamped;
        setCropBox(clamped);
      },

      onPanResponderRelease: () => {
        activeHandle.current = null;
        startCrop.current = null;
      },
    })
  ).current;

  // Apply crop and navigate to Preview
  const handleCrop = async () => {
    if (!cropBox || !imageLayout || !imageSize) return;

    try {
      const scaleX = imageSize.width / imageLayout.width;
      const scaleY = imageSize.height / imageLayout.height;

      const originX = Math.round(cropBox.x * scaleX);
      const originY = Math.round(cropBox.y * scaleY);
      const cropWidth = Math.round(cropBox.width * scaleX);
      const cropHeight = Math.round(cropBox.height * scaleY);

      const result = await ImageManipulator.manipulateAsync(
        imageUri,
        [
          {
            crop: {
              originX: Math.max(0, originX),
              originY: Math.max(0, originY),
              width: Math.min(cropWidth, imageSize.width - originX),
              height: Math.min(cropHeight, imageSize.height - originY),
            },
          },
        ],
        { compress: 0.9, format: ImageManipulator.SaveFormat.JPEG }
      );

      navigation.replace('Preview', { imageUri: result.uri });
    } catch (error) {
      console.error('Crop failed:', error);
      Alert.alert('Crop Failed', 'Could not crop the image. Using original.');
      navigation.replace('Preview', { imageUri });
    }
  };

  // Skip crop, use full image
  const handleSkip = () => {
    navigation.replace('Preview', { imageUri });
  };

  if (!imageLayout || !cropBox) {
    return (
      <View style={styles.container}>
        <Text style={styles.loadingText}>Loading image...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Top bar */}
      <SafeAreaView style={styles.topBar} edges={['top']}>
        <Text style={styles.topBarTitle}>Crop Document</Text>
        <Text style={styles.topBarSubtitle}>Drag corners to trim dead space</Text>
      </SafeAreaView>

      {/* Image + crop overlay */}
      <View
        style={[
          styles.imageContainer,
          {
            width: imageLayout.width,
            height: imageLayout.height,
            marginTop: imageLayout.offsetY,
          },
        ]}
        {...panResponder.panHandlers}
      >
        {/* The image */}
        <Image
          source={{ uri: imageUri }}
          style={{ width: imageLayout.width, height: imageLayout.height }}
          resizeMode="contain"
        />

        {/* Dimmed overlay outside crop area (4 rectangles) */}
        <View
          style={[styles.dimOverlay, { top: 0, left: 0, right: 0, height: cropBox.y }]}
        />
        <View
          style={[
            styles.dimOverlay,
            { top: cropBox.y + cropBox.height, left: 0, right: 0, bottom: 0 },
          ]}
        />
        <View
          style={[
            styles.dimOverlay,
            { top: cropBox.y, left: 0, width: cropBox.x, height: cropBox.height },
          ]}
        />
        <View
          style={[
            styles.dimOverlay,
            {
              top: cropBox.y,
              left: cropBox.x + cropBox.width,
              right: 0,
              height: cropBox.height,
            },
          ]}
        />

        {/* Crop border */}
        <View
          style={[
            styles.cropBorder,
            {
              left: cropBox.x,
              top: cropBox.y,
              width: cropBox.width,
              height: cropBox.height,
            },
          ]}
        >
          {/* Grid lines */}
          <View style={[styles.gridLine, styles.gridH, { top: '33.3%' }]} />
          <View style={[styles.gridLine, styles.gridH, { top: '66.6%' }]} />
          <View style={[styles.gridLine, styles.gridV, { left: '33.3%' }]} />
          <View style={[styles.gridLine, styles.gridV, { left: '66.6%' }]} />
        </View>

        {/* Corner handles */}
        <View
          style={[
            styles.handle,
            styles.cornerTL,
            { left: cropBox.x - HANDLE_SIZE / 2, top: cropBox.y - HANDLE_SIZE / 2 },
          ]}
        />
        <View
          style={[
            styles.handle,
            styles.cornerTR,
            {
              left: cropBox.x + cropBox.width - HANDLE_SIZE / 2,
              top: cropBox.y - HANDLE_SIZE / 2,
            },
          ]}
        />
        <View
          style={[
            styles.handle,
            styles.cornerBL,
            {
              left: cropBox.x - HANDLE_SIZE / 2,
              top: cropBox.y + cropBox.height - HANDLE_SIZE / 2,
            },
          ]}
        />
        <View
          style={[
            styles.handle,
            styles.cornerBR,
            {
              left: cropBox.x + cropBox.width - HANDLE_SIZE / 2,
              top: cropBox.y + cropBox.height - HANDLE_SIZE / 2,
            },
          ]}
        />
      </View>

      {/* Bottom actions */}
      <SafeAreaView style={styles.bottomBar} edges={['bottom']}>
        <TouchableOpacity style={styles.skipButton} onPress={handleSkip}>
          <Text style={styles.skipButtonText}>Skip</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.cropButton} onPress={handleCrop}>
          <Ionicons name="crop-outline" size={20} color="#fff" />
          <Text style={styles.cropButtonText}>Crop & Continue</Text>
        </TouchableOpacity>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
  },
  loadingText: {
    color: '#9ca3af',
    fontSize: 16,
    marginTop: SCREEN_HEIGHT / 2 - 20,
  },
  topBar: {
    width: '100%',
    paddingHorizontal: 16,
    paddingBottom: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    alignItems: 'center',
  },
  topBarTitle: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '600',
  },
  topBarSubtitle: {
    color: '#9ca3af',
    fontSize: 13,
    marginTop: 2,
  },
  imageContainer: {
    position: 'relative',
  },
  dimOverlay: {
    position: 'absolute',
    backgroundColor: 'rgba(0, 0, 0, 0.55)',
  },
  cropBorder: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#0ea5e9',
  },
  gridLine: {
    position: 'absolute',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
  },
  gridH: {
    left: 0,
    right: 0,
    height: 1,
  },
  gridV: {
    top: 0,
    bottom: 0,
    width: 1,
  },
  handle: {
    position: 'absolute',
    width: HANDLE_SIZE,
    height: HANDLE_SIZE,
    borderColor: '#0ea5e9',
    borderWidth: 3,
    backgroundColor: 'rgba(14, 165, 233, 0.15)',
  },
  cornerTL: {
    borderRightWidth: 0,
    borderBottomWidth: 0,
    borderTopLeftRadius: 4,
  },
  cornerTR: {
    borderLeftWidth: 0,
    borderBottomWidth: 0,
    borderTopRightRadius: 4,
  },
  cornerBL: {
    borderRightWidth: 0,
    borderTopWidth: 0,
    borderBottomLeftRadius: 4,
  },
  cornerBR: {
    borderLeftWidth: 0,
    borderTopWidth: 0,
    borderBottomRightRadius: 4,
  },
  bottomBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 12,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  skipButton: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
    backgroundColor: '#374151',
  },
  skipButtonText: {
    color: '#d1d5db',
    fontSize: 16,
    fontWeight: '500',
  },
  cropButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
    backgroundColor: '#0ea5e9',
  },
  cropButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
