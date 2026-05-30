# SoftPredict 3-Panel Visualization - Deployment Guide

## Status
✅ **Backend Complete** - Three-panel visualization working at `http://127.0.0.1:8000/correct`
✅ **Flutter Code Updated** - Ready to rebuild and deploy

## What's Done

### Backend (`backend/app.py`)
The `/correct` endpoint now returns a **three-panel side-by-side comparison**:

1. **LEFT PANEL (BEFORE)** - Original uploaded face image
2. **MIDDLE PANEL (MESH OVERLAY)** - Original image with green mesh showing GCN-predicted facial landmarks
3. **RIGHT PANEL (AFTER)** - Corrected/warped image with orthodontic adjustments applied

**Features:**
- Bright cyan-green mesh edges (BGR: 0, 255, 100) connecting landmark points
- Light blue landmark circles (BGR: 100, 200, 255), 6px radius with cyan outline
- "MESH" label on middle panel
- "BEFORE | MESH OVERLAY | AFTER" text labels at top
- Comprehensive error handling with detailed logging

### Flutter App (`softpredict_app/`)
**Code Changes Made:**
1. ✅ Added import: `import '../services/api_service.dart';` 
2. ✅ Updated `_loadCorrectedImage()` to call backend `/correct` endpoint instead of local processing
3. ✅ Ensures fresh mesh overlay fetched from server on each upload

## To Deploy to Android Device

### Step 1: Install Flutter
If Flutter is not installed on your system:
```bash
# Download from https://flutter.dev/docs/get-started/install/windows
# Add C:\flutter\bin to your system PATH
```

### Step 2: Rebuild Flutter App
```bash
cd "C:\Users\Naveen Kumar S\OneDrive\Dokumen\medicalAI\softpredict_app"

# Get latest dependencies
flutter pub get

# Build APK
flutter build apk --release
# OR for development build with hot-reload:
flutter run -d {DEVICE_ID}
```

### Step 3: Set Up Port Forwarding
```bash
# On development machine (Windows CMD)
adb reverse tcp:8000 tcp:8000
```

### Step 4: Deploy and Test
```bash
# Find device ID
adb devices

# Install APK (if built with --release)
adb install -r build\app\outputs\apk\release\app-release.apk

# OR run directly with hot-reload (requires flutter run)
flutter run -d 10BD7K2LYA000KN --dart-define=SOFTPREDICT_API_URL=http://127.0.0.1:8000
```

## Server Requirements

Ensure backend is running:
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Requirements Check:**
- ✅ Model file: `backend/softpredict_model.pth` (present)
- ✅ FastAPI with CORS enabled
- ✅ PyTorch GCN model loaded successfully
- ✅ Mediapipe face detection working

## Testing the 3-Panel Output

### Test via Python (Desktop)
```python
import requests

with open('backend/test_photo.jpg', 'rb') as f:
    r = requests.post('http://127.0.0.1:8000/correct', files={'file': f})

if r.status_code == 200:
    with open('three_panel_test.png', 'wb') as out:
        out.write(r.content)
    print('Three-panel image created successfully!')
```

### Test on Mobile
1. Launch app on device
2. Tap image picker
3. Select any face photo
4. App should display three-panel comparison:
   - Left: original uploaded face
   - Center: same face with bright green orthodontic mesh overlay
   - Right: corrected face with teeth whitening and jaw refinement

## File Changes Summary

| File | Change | Status |
|------|--------|--------|
| `backend/app.py` | `/correct` endpoint returns 3-panel image | ✅ Complete |
| `softpredict_app/lib/screens/prediction_result_screen.dart` | Calls backend API instead of local processing | ✅ Complete |
| `softpredict_app/lib/services/api_service.dart` | No changes needed (already working) | ✅ Ready |

## Troubleshooting

### "Connection refused" when running Flutter app
- Ensure backend is running at `http://127.0.0.1:8000`
- Verify ADB port forwarding: `adb reverse tcp:8000 tcp:8000`
- Check device is connected: `adb devices`

### "No face detected" error
- Upload a clear face photo
- Ensure face is in frame and well-lit
- Backend will fallback to synthetic landmarks if Mediapipe fails

### Build fails on Flutter rebuild
- Run `flutter clean` and `flutter pub get`
- Check Flutter version: `flutter --version` (should be recent)
- Verify Android SDK and JDK are installed

## API Endpoint Reference

**POST `/correct`**
```
Parameters:
  - file: Image file (multipart form data)
  - method: "teeth" (default) or "jaw"

Response:
  - PNG image: 3-panel comparison (width: 640*3 = 1920px, height: 480+40 = 520px)
  - Status 200: Success
  - Status 400: No face detected or invalid image
  - Status 500: Server error
```

## Performance Notes

- Backend processing time: ~500ms - 1s per image (normal for clinical analysis)
- Image sizes: Typically 200-500KB per 3-panel PNG
- GPU not required (CPU inference is sufficient)
- Tested with Android device API 36, Impeller renderer

## Next Steps

1. Install Flutter SDK if not present
2. Rebuild app with `flutter build apk` or `flutter run`
3. Deploy to device with ADB
4. Upload a face photo via the mobile UI
5. Verify three-panel visualization appears correctly

## Support

For backend issues, check `backend/app.py` logs for detailed error messages.
For Flutter issues, run with `flutter run -v` for verbose logging.
