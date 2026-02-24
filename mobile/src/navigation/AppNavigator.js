// ============================================================================
// App Navigation
// ============================================================================

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';

// Screens
import CameraScreen from '../screens/CameraScreen';
import CropScreen from '../screens/CropScreen';
import PreviewScreen from '../screens/PreviewScreen';
import UploadsScreen from '../screens/UploadsScreen';
import JobDetailScreen from '../screens/JobDetailScreen';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

// ============================================================================
// Capture Stack (Camera + Preview)
// ============================================================================

function CaptureStack() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: '#0ea5e9',
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: '600',
        },
      }}
    >
      <Stack.Screen
        name="Camera"
        component={CameraScreen}
        options={{
          title: 'Capture Document',
          headerShown: false,
        }}
      />
      <Stack.Screen
        name="Crop"
        component={CropScreen}
        options={{
          title: 'Crop Document',
          headerShown: false,
        }}
      />
      <Stack.Screen
        name="Preview"
        component={PreviewScreen}
        options={{
          title: 'Review & Upload',
          presentation: 'modal',
        }}
      />
    </Stack.Navigator>
  );
}

// ============================================================================
// Uploads Stack (List + Detail)
// ============================================================================

function UploadsStack() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: '#0ea5e9',
        },
        headerTintColor: '#fff',
        headerTitleStyle: {
          fontWeight: '600',
        },
      }}
    >
      <Stack.Screen
        name="UploadsList"
        component={UploadsScreen}
        options={{ title: 'Recent Uploads' }}
      />
      <Stack.Screen
        name="JobDetail"
        component={JobDetailScreen}
        options={{ title: 'Processing Details' }}
      />
    </Stack.Navigator>
  );
}

// ============================================================================
// Main Tab Navigator
// ============================================================================

export default function AppNavigator() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName;

            if (route.name === 'Capture') {
              iconName = focused ? 'camera' : 'camera-outline';
            } else if (route.name === 'Uploads') {
              iconName = focused ? 'documents' : 'documents-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#0ea5e9',
          tabBarInactiveTintColor: 'gray',
          headerShown: false,
          tabBarStyle: {
            backgroundColor: '#fff',
            borderTopColor: '#e5e7eb',
            paddingBottom: 5,
            paddingTop: 5,
            height: 60,
          },
          tabBarLabelStyle: {
            fontSize: 12,
            fontWeight: '500',
          },
        })}
      >
        <Tab.Screen
          name="Capture"
          component={CaptureStack}
          options={{ title: 'Capture' }}
        />
        <Tab.Screen
          name="Uploads"
          component={UploadsStack}
          options={{ title: 'Uploads' }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
