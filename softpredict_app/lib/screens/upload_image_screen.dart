import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/face_prediction.dart';
import '../services/api_service.dart';
import 'loading_screen.dart';
import 'prediction_result_screen.dart';

class UploadImageScreen extends StatefulWidget {
  const UploadImageScreen({super.key});

  @override
  State<UploadImageScreen> createState() => _UploadImageScreenState();
}

class _UploadImageScreenState extends State<UploadImageScreen> {
  final ApiService _apiService = ApiService();
  final ImagePicker _imagePicker = ImagePicker();

  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  String _statusText = 'Upload a face image to start the demo.';

  Future<void> _pickImage() async {
    final picked = await _imagePicker.pickImage(source: ImageSource.gallery);
    if (picked == null) {
      return;
    }

    final bytes = await picked.readAsBytes();
    setState(() {
      _selectedImageBytes = bytes;
      _selectedImageName = picked.name;
      _statusText = 'Image selected. Ready to analyze.';
    });
  }

  Future<void> _runPrediction() async {
    final imageBytes = _selectedImageBytes;
    final imageName = _selectedImageName ?? 'upload.jpg';
    if (imageBytes == null) {
      setState(() {
        _statusText = 'Please select an image first.';
      });
      return;
    }

    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const LoadingScreen()),
    );

    try {
      final FacePrediction prediction = await _apiService.predictImageBytes(imageBytes, imageName);
      if (!mounted) {
        return;
      }

      Navigator.of(context).pop();
      Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => PredictionResultScreen(
            imageBytes: imageBytes,
            prediction: prediction,
          ),
        ),
      );
    } catch (error) {
      if (!mounted) {
        return;
      }

      Navigator.of(context).pop();
      setState(() {
        _statusText = 'Prediction failed: $error';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Upload Face Image')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: Card(
                elevation: 2,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text(
                        'Simple Demo Flow',
                        style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 12),
                      const Text('1. Upload image'),
                      const Text('2. Send to FastAPI backend'),
                      const Text('3. Extract landmarks'),
                      const Text('4. Show before/after overlay'),
                      const SizedBox(height: 16),
                      Expanded(
                        child: Container(
                          decoration: BoxDecoration(
                            color: const Color(0xFFF8FAFC),
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(color: const Color(0xFFE2E8F0)),
                          ),
                          clipBehavior: Clip.antiAlias,
                            child: _selectedImageBytes == null
                              ? const Center(
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(Icons.image_outlined, size: 56, color: Colors.black38),
                                      SizedBox(height: 12),
                                      Text('No image selected'),
                                    ],
                                  ),
                                )
                              : Image.memory(_selectedImageBytes!, fit: BoxFit.cover),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              _statusText,
              textAlign: TextAlign.center,
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _pickImage,
              icon: const Icon(Icons.upload_file),
              label: const Text('Upload Image'),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: _runPrediction,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Run Prediction'),
            ),
          ],
        ),
      ),
    );
  }
}
