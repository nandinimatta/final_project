import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;

import '../models/face_prediction.dart';

class ApiService {
  static String get baseUrl {
    // Use localhost for web (browser), 10.0.2.2 for Android emulator
    final env = const String.fromEnvironment('SOFTPREDICT_API_URL');
    if (env.isNotEmpty) return env;
    if (kIsWeb) return 'http://localhost:8000';
    return 'http://10.0.2.2:8000';
  }

  // Legacy file-path based upload removed to keep web compatibility.
  // Use `predictImageBytes` which works on mobile and web.

  /// Send raw image bytes to backend. Use this for web compatibility
  Future<FacePrediction> predictImageBytes(Uint8List bytes, String filename) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    // Ensure the client closes connection after request to avoid partial-header issues on Android
    request.headers['Connection'] = 'close';

    http.StreamedResponse response;
    try {
      response = await request.send().timeout(const Duration(seconds: 60));
    } catch (e) {
      throw Exception('Network error while contacting server: $e');
    }
    final body = await response.stream.bytesToString();

    if (response.statusCode >= 400) {
      final errorMessage = _extractErrorMessage(body);
      throw Exception(errorMessage);
    }

    final decoded = jsonDecode(body) as Map<String, dynamic>;
    return FacePrediction.fromJson(decoded);
  }

  Future<Map<String, Uint8List>> correctImageThreePanels(
    Uint8List bytes,
    String filename, {
    String method = 'jaw',
  }) async {
    final uri = Uri.parse('$baseUrl/correct?method=$method');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    // Ensure the client closes connection after request to avoid partial-header issues on Android
    request.headers['Connection'] = 'close';

    http.StreamedResponse response;
    try {
      response = await request.send().timeout(const Duration(seconds: 90));
    } catch (e) {
      throw Exception('Network error while contacting server: $e');
    }
    final responseBytes = await response.stream.toBytes();

    if (response.statusCode >= 400) {
      final body = utf8.decode(responseBytes, allowMalformed: true);
      final errorMessage = _extractErrorMessage(body);
      throw Exception(errorMessage);
    }

    try {
      final body = utf8.decode(responseBytes);
      print('[API] Response body (first 200 chars): ${body.substring(0, min(200, body.length))}');
      
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      print('[API] Decoded JSON keys: ${decoded.keys.toList()}');
      
      final beforeB64 = (decoded['before'] as String).replaceFirst('data:image/png;base64,', '');
      final meshB64 = (decoded['mesh'] as String).replaceFirst('data:image/png;base64,', '');
      final afterB64 = (decoded['after'] as String).replaceFirst('data:image/png;base64,', '');
      
      print('[API] Before B64 length: ${beforeB64.length}');
      print('[API] Mesh B64 length: ${meshB64.length}');
      print('[API] After B64 length: ${afterB64.length}');
      
      return {
        'before': base64Decode(beforeB64),
        'mesh': base64Decode(meshB64),
        'after': base64Decode(afterB64),
      };
    } catch (e) {
      print('[API] Error parsing response: $e');
      print('[API] Response status: ${response.statusCode}');
      print('[API] Response bytes (first 200): ${responseBytes.take(200).toList()}');
      rethrow;
    }
  }

  Future<Uint8List> correctImageBytes(
    Uint8List bytes,
    String filename, {
    String method = 'jaw',
  }) async {
    final uri = Uri.parse('$baseUrl/correct?method=$method');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    // Ensure the client closes connection after request to avoid partial-header issues on Android
    request.headers['Connection'] = 'close';

    http.StreamedResponse response;
    try {
      response = await request.send().timeout(const Duration(seconds: 90));
    } catch (e) {
      throw Exception('Network error while contacting server: $e');
    }
    final responseBytes = await response.stream.toBytes();

    if (response.statusCode >= 400) {
      final body = utf8.decode(responseBytes, allowMalformed: true);
      final errorMessage = _extractErrorMessage(body);
      throw Exception(errorMessage);
    }

    return Uint8List.fromList(responseBytes);
  }

  Future<Map<String, dynamic>> login(String username, String password) async {
    final uri = Uri.parse('$baseUrl/login');
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'password': password}),
    ).timeout(const Duration(seconds: 15));
    if (response.statusCode >= 400) {
      final errorMessage = _extractErrorMessage(response.body);
      throw Exception(errorMessage);
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<List<dynamic>> getRecords() async {
    final uri = Uri.parse('$baseUrl/records');
    final response = await http.get(uri).timeout(const Duration(seconds: 20));
    if (response.statusCode >= 400) {
      final errorMessage = _extractErrorMessage(response.body);
      throw Exception(errorMessage);
    }
    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    return decoded['records'] as List<dynamic>;
  }

  Future<Map<String, dynamic>> createRecord({
    required Uint8List bytes,
    required String filename,
    required String patientId,
    required String name,
    required int age,
    required String dob,
    required String gender,
    required String problem,
    required String treatmentMethod,
  }) async {
    final uri = Uri.parse('$baseUrl/records/create');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    request.fields['patient_id'] = patientId;
    request.fields['name'] = name;
    request.fields['age'] = age.toString();
    request.fields['dob'] = dob;
    request.fields['gender'] = gender;
    request.fields['problem'] = problem;
    request.fields['treatment_method'] = treatmentMethod;
    request.headers['Connection'] = 'close';

    http.StreamedResponse response;
    try {
      response = await request.send().timeout(const Duration(seconds: 90));
    } catch (e) {
      throw Exception('Network error while contacting server: $e');
    }
    final body = await response.stream.bytesToString();

    if (response.statusCode >= 400) {
      final errorMessage = _extractErrorMessage(body);
      throw Exception(errorMessage);
    }

    return jsonDecode(body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> register(String username, String email, String password) async {
    final uri = Uri.parse('$baseUrl/register');
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'email': email, 'password': password}),
    ).timeout(const Duration(seconds: 15));
    if (response.statusCode >= 400) {
      final errorMessage = _extractErrorMessage(response.body);
      throw Exception(errorMessage);
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<void> deleteRecord(int id) async {
    final uri = Uri.parse('$baseUrl/records/$id');
    final response = await http.delete(uri).timeout(const Duration(seconds: 15));
    if (response.statusCode >= 400) {
      final errorMessage = _extractErrorMessage(response.body);
      throw Exception(errorMessage);
    }
  }

  String _extractErrorMessage(String body) {
    try {
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      return decoded['detail']?.toString() ?? 'Prediction failed.';
    } catch (_) {
      return 'Prediction failed.';
    }
  }
}
