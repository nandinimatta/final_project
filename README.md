# SoftPredict Laptop Setup

This workspace is split into two parts:

- `backend/` for the FastAPI inference server on the laptop
- `softpredict_app/` for the Flutter mobile app

The workflow is intentionally simple for a 2-day demo:

1. Install Flutter SDK and Android Studio on the laptop.
2. Install the Python backend libraries.
3. Train the small model in Google Colab and export `softpredict_model.pth`.
4. Copy that `.pth` file into `backend/` on the laptop.
5. Run the FastAPI backend and connect the Flutter app to it.

## What this scaffold includes

- Splash Screen
- Upload Image Screen
- Loading Screen
- Prediction Result Screen
- FastAPI endpoint for image upload and prediction
- MediaPipe face landmark extraction
- GCN-style placeholder model that can load your Colab weights
- Demo fallback that simulates soft-tissue movement if the model file is not present yet

## Laptop install commands

### Flutter and Android Studio

Install Flutter SDK and Android Studio on the laptop first.

The Flutter source is already scaffolded in `softpredict_app/`. If you want Flutter to add the Android and iOS platform folders locally, open that folder after installing Flutter and run `flutter create .` only if you need the generated shell.

Keep the package name as `softpredict_app` so the source files and pubspec match.

### Backend libraries

```powershell
cd "c:\Users\Naveen Kumar S\OneDrive\Dokumen\medicalAI\backend"
pip install fastapi uvicorn torch mediapipe opencv-python python-multipart numpy
```

## Run backend

```powershell
cd "c:\Users\Naveen Kumar S\OneDrive\Dokumen\medicalAI\backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Mobile Setup & USB Testing (5G Cellular & USB Cable)

1. **Enable Windows Developer Mode**:
   Go to `Windows Settings` -> `Privacy & Security` -> `For Developers` -> Turn **ON** Developer Mode.

2. **USB Port Forwarding (ADB Reverse)**:
   When running the mobile app over USB cable (especially when mobile is connected to 5G cellular data), run ADB reverse to route requests to your laptop backend:
   ```powershell
   adb reverse tcp:8000 tcp:8000
   ```

3. **Backend Authentication Details**:
   - Default Doctor Username: `doctor123`
   - Default Doctor Password: `password123`

You can pass the URL with:

```powershell
flutter run --dart-define=SOFTPREDICT_API_URL=http://127.0.0.1:8000
```

## Colab to laptop handoff

Train in Colab and save the model with the same architecture used in `backend/model.py`.

Recommended export:

```python
torch.save(model.state_dict(), "softpredict_model.pth")
```

Then copy `softpredict_model.pth` into `backend/` on the laptop.

## Demo flow

1. Open the app.
2. Upload a face image.
3. Show the loading animation.
4. Backend extracts landmarks.
5. Backend predicts slight landmark movement.
6. Result screen shows before/after mesh overlay.
