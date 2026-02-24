# Medical Ingestion Mobile App

A React Native mobile app built with Expo for capturing and uploading medical documents.

## Features

- **Camera Capture**: Take photos of medical documents with a guided frame
- **Gallery Import**: Select existing images from device gallery
- **Document Type Selection**: Auto-detect or manually specify document type
- **Quality Feedback**: Real-time image quality warnings with tips
- **Upload Tracking**: View all uploads with status (pending, processing, completed, failed)
- **Result Viewing**: See extracted values, confidence scores, and processing steps

## Setup

### Prerequisites

- Node.js 18+
- Expo CLI: `npm install -g expo-cli`
- Expo Go app on your phone (for device testing)
- Backend server running on port 8000

### Installation

```bash
cd mobile
npm install
```

### Configuration

For device testing, update the API base URL in `src/services/api.js`:

```javascript
// Replace localhost with your machine's IP address
const API_BASE = 'http://192.168.1.XXX:8000';
```

Find your IP address:
- macOS: `ifconfig | grep "inet " | grep -v 127.0.0.1`
- Windows: `ipconfig`
- Linux: `hostname -I`

### Running

```bash
# Start Expo development server
npm start

# Or run on specific platform
npm run ios
npm run android
```

Then scan the QR code with Expo Go (Android) or Camera app (iOS).

## Project Structure

```
mobile/
├── App.js                 # App entry point
├── app.json               # Expo configuration
├── package.json           # Dependencies
├── src/
│   ├── navigation/
│   │   └── AppNavigator.js    # Tab and stack navigation
│   ├── screens/
│   │   ├── CameraScreen.js    # Document capture
│   │   ├── PreviewScreen.js   # Review and upload
│   │   ├── UploadsScreen.js   # Recent uploads list
│   │   └── JobDetailScreen.js # Processing results
│   ├── services/
│   │   └── api.js             # Backend API calls
│   └── components/            # Reusable components
└── assets/                    # App icons and splash
```

## API Endpoints Used

- `POST /api/upload` - Upload document image
- `POST /api/upload-and-analyze` - Upload with quality analysis
- `POST /api/process` - Start document processing
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{id}` - Get job status and results
- `DELETE /api/jobs/{id}` - Delete a job

## Troubleshooting

### "Network request failed"
- Ensure backend is running: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Check API_BASE URL uses your machine's local IP (not localhost)
- Ensure phone is on the same WiFi network as your computer

### Camera not working
- Grant camera permissions when prompted
- On iOS, check Settings > Privacy > Camera
- On Android, check Settings > Apps > Medical Ingestion > Permissions

### Image quality errors
- Ensure good lighting
- Hold camera steady and close to document
- Make sure document is flat and in focus
- Minimum resolution: 600x450 pixels
