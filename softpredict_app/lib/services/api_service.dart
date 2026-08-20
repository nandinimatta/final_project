import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb, debugPrint;

import '../models/face_prediction.dart';

class ApiService {
  static String? _cachedWorkingBaseUrl;

  static List<String> get candidateUrls {
    final env = const String.fromEnvironment('SOFTPREDICT_API_URL');
    final list = <String>[];
    if (env.isNotEmpty) {
      list.add(env);
    }
    if (kIsWeb) {
      list.addAll([
        'http://localhost:8000',
        'http://127.0.0.1:8000',
        'http://10.182.25.121:8000',
        'http://172.23.19.60:8000',
        'http://172.23.18.177:8000',
      ]);
    } else {
      list.addAll([
        'http://127.0.0.1:8000',
        'http://10.182.25.121:8000',
        'http://172.23.19.60:8000',
        'http://172.23.18.177:8000',
        'http://10.0.2.2:8000',
        'http://localhost:8000',
      ]);
    }
    return list.toSet().toList();
  }

  static String get baseUrl {
    return _cachedWorkingBaseUrl ?? candidateUrls.first;
  }

  static bool _isServerInfrastructureError(int statusCode, String body) {
    if (statusCode >= 502) return true;
    final lower = body.toLowerCase();
    if (lower.contains('space is in error') ||
        lower.contains('check its status on hf.co') ||
        lower.contains('503 service unavailable')) {
      return true;
    }
    return false;
  }

  /// Helper to send JSON POST request with automatic server URL failover and 120s timeout
  Future<http.Response> _postJsonWithFailover(
    String endpointPath,
    Map<String, dynamic> body, {
    Duration timeout = const Duration(seconds: 120),
  }) async {
    final urlsToTry = <String>[];
    if (_cachedWorkingBaseUrl != null) {
      urlsToTry.add(_cachedWorkingBaseUrl!);
    }
    for (final url in candidateUrls) {
      if (!urlsToTry.contains(url)) {
        urlsToTry.add(url);
      }
    }

    Object? lastError;
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base$endpointPath');
        debugPrint('[API] Trying POST $uri');
        final response = await http.post(
          uri,
          headers: {'Content-Type': 'application/json', 'Connection': 'close'},
          body: jsonEncode(body),
        ).timeout(timeout);

        if (_isServerInfrastructureError(response.statusCode, response.body)) {
          if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
          lastError = 'Host returned status ${response.statusCode}';
          debugPrint('[API] Server $base returned host error (${response.statusCode}). Trying next...');
          continue;
        }

        _cachedWorkingBaseUrl = base;
        debugPrint('[API] Connected successfully to $base');
        return response;
      } catch (e) {
        if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
        lastError = e;
        debugPrint('[API] Connection to $base$endpointPath failed: $e. Trying next server URL...');
      }
    }
    throw Exception('Connection failed to backend server across candidate endpoints ($urlsToTry). Details: $lastError');
  }

  /// Helper to send GET request with automatic server URL failover and 120s timeout
  Future<http.Response> _getWithFailover(
    String endpointPath, {
    Duration timeout = const Duration(seconds: 120),
  }) async {
    final urlsToTry = <String>[];
    if (_cachedWorkingBaseUrl != null) {
      urlsToTry.add(_cachedWorkingBaseUrl!);
    }
    for (final url in candidateUrls) {
      if (!urlsToTry.contains(url)) {
        urlsToTry.add(url);
      }
    }

    Object? lastError;
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base$endpointPath');
        debugPrint('[API] Trying GET $uri');
        final response = await http.get(
          uri,
          headers: {'Connection': 'close'},
        ).timeout(timeout);

        if (_isServerInfrastructureError(response.statusCode, response.body)) {
          if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
          lastError = 'Host returned status ${response.statusCode}';
          debugPrint('[API] Server $base returned host error (${response.statusCode}). Trying next...');
          continue;
        }

        _cachedWorkingBaseUrl = base;
        return response;
      } catch (e) {
        if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
        lastError = e;
        debugPrint('[API] GET to $base$endpointPath failed: $e. Trying next server URL...');
      }
    }
    throw Exception('Connection failed to backend server across candidate endpoints ($urlsToTry). Details: $lastError');
  }

  Future<FacePrediction> predictImageBytes(Uint8List bytes, String filename) async {
    final urlsToTry = <String>[];
    if (_cachedWorkingBaseUrl != null) urlsToTry.add(_cachedWorkingBaseUrl!);
    for (final url in candidateUrls) {
      if (!urlsToTry.contains(url)) urlsToTry.add(url);
    }

    Object? lastError;
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base/predict');
        final request = http.MultipartRequest('POST', uri);
        request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
        request.headers['Connection'] = 'close';

        final response = await request.send().timeout(const Duration(seconds: 120));
        final body = await response.stream.bytesToString();

        if (_isServerInfrastructureError(response.statusCode, body)) {
          if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
          lastError = 'Host returned status ${response.statusCode}';
          continue;
        }

        if (response.statusCode >= 400) {
          final errorMessage = _extractErrorMessage(body);
          throw Exception(errorMessage);
        }

        _cachedWorkingBaseUrl = base;
        final decoded = jsonDecode(body) as Map<String, dynamic>;
        return FacePrediction.fromJson(decoded);
      } catch (e) {
        if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
        lastError = e;
        debugPrint('[API] Multipart predict to $base failed: $e');
      }
    }
    throw Exception('Network error while contacting server: $lastError');
  }

  Future<Map<String, Uint8List>> correctImageThreePanels(
    Uint8List bytes,
    String filename, {
    String method = 'jaw',
  }) async {
    final urlsToTry = <String>[];
    if (_cachedWorkingBaseUrl != null) urlsToTry.add(_cachedWorkingBaseUrl!);
    for (final url in candidateUrls) {
      if (!urlsToTry.contains(url)) urlsToTry.add(url);
    }

    Object? lastError;
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base/correct?method=$method');
        final request = http.MultipartRequest('POST', uri);
        request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
        request.headers['Connection'] = 'close';

        final response = await request.send().timeout(const Duration(seconds: 120));
        final responseBytes = await response.stream.toBytes();
        final bodyText = utf8.decode(responseBytes, allowMalformed: true);

        if (_isServerInfrastructureError(response.statusCode, bodyText)) {
          if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
          lastError = 'Host returned status ${response.statusCode}';
          continue;
        }

        if (response.statusCode >= 400) {
          final errorMessage = _extractErrorMessage(bodyText);
          throw Exception(errorMessage);
        }

        _cachedWorkingBaseUrl = base;
        final decoded = jsonDecode(bodyText) as Map<String, dynamic>;

        final beforeB64 = (decoded['before'] as String).replaceFirst('data:image/png;base64,', '');
        final meshB64 = (decoded['mesh'] as String).replaceFirst('data:image/png;base64,', '');
        final afterB64 = (decoded['after'] as String).replaceFirst('data:image/png;base64,', '');

        return {
          'before': base64Decode(beforeB64),
          'mesh': base64Decode(meshB64),
          'after': base64Decode(afterB64),
        };
      } catch (e) {
        if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
        lastError = e;
        debugPrint('[API] Multipart correct to $base failed: $e');
      }
    }
    throw Exception('Network error while contacting server: $lastError');
  }

  Future<Uint8List> correctImageBytes(
    Uint8List bytes,
    String filename, {
    String method = 'jaw',
  }) async {
    final urlsToTry = <String>[];
    if (_cachedWorkingBaseUrl != null) urlsToTry.add(_cachedWorkingBaseUrl!);
    for (final url in candidateUrls) {
      if (!urlsToTry.contains(url)) urlsToTry.add(url);
    }

    Object? lastError;
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base/correct?method=$method');
        final request = http.MultipartRequest('POST', uri);
        request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
        request.headers['Connection'] = 'close';

        final response = await request.send().timeout(const Duration(seconds: 120));
        final responseBytes = await response.stream.toBytes();
        final bodyText = utf8.decode(responseBytes, allowMalformed: true);

        if (_isServerInfrastructureError(response.statusCode, bodyText)) {
          if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
          lastError = 'Host returned status ${response.statusCode}';
          continue;
        }

        if (response.statusCode >= 400) {
          final errorMessage = _extractErrorMessage(bodyText);
          throw Exception(errorMessage);
        }

        _cachedWorkingBaseUrl = base;
        return Uint8List.fromList(responseBytes);
      } catch (e) {
        if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
        lastError = e;
        debugPrint('[API] Multipart correctImageBytes to $base failed: $e');
      }
    }
    throw Exception('Network error while contacting server: $lastError');
  }

  Future<Map<String, dynamic>> login(String username, String password) async {
    final response = await _postJsonWithFailover(
      '/login',
      {'username': username, 'password': password},
      timeout: const Duration(seconds: 120),
    );

    if (response.statusCode >= 400) {
      final errorMessage = _extractErrorMessage(response.body);
      throw Exception(errorMessage);
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<List<dynamic>> getRecords({String? doctorUsername}) async {
    String endpointPath = '/records';
    if (doctorUsername != null && doctorUsername.trim().isNotEmpty) {
      endpointPath += '?doctor_username=${Uri.encodeComponent(doctorUsername.trim())}';
    }
    final response = await _getWithFailover(
      endpointPath,
      timeout: const Duration(seconds: 120),
    );

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
    String? doctorUsername,
  }) async {
    final urlsToTry = <String>[];
    if (_cachedWorkingBaseUrl != null) urlsToTry.add(_cachedWorkingBaseUrl!);
    for (final url in candidateUrls) {
      if (!urlsToTry.contains(url)) urlsToTry.add(url);
    }

    Object? lastError;
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base/records/create');
        final request = http.MultipartRequest('POST', uri);
        request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
        request.fields['patient_id'] = patientId;
        request.fields['name'] = name;
        request.fields['age'] = age.toString();
        request.fields['dob'] = dob;
        request.fields['gender'] = gender;
        request.fields['problem'] = problem;
        request.fields['treatment_method'] = treatmentMethod;
        if (doctorUsername != null && doctorUsername.trim().isNotEmpty) {
          request.fields['doctor_username'] = doctorUsername.trim();
        }
        request.headers['Connection'] = 'close';

        final response = await request.send().timeout(const Duration(seconds: 120));
        final body = await response.stream.bytesToString();

        if (_isServerInfrastructureError(response.statusCode, body)) {
          if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
          lastError = 'Host returned status ${response.statusCode}';
          continue;
        }

        if (response.statusCode >= 400) {
          final errorMessage = _extractErrorMessage(body);
          throw Exception(errorMessage);
        }

        _cachedWorkingBaseUrl = base;
        return jsonDecode(body) as Map<String, dynamic>;
      } catch (e) {
        if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
        lastError = e;
        debugPrint('[API] createRecord to $base failed: $e');
      }
    }
    throw Exception('Network error while contacting server: $lastError');
  }

  Future<Map<String, dynamic>> register(String username, String email, String password) async {
    final response = await _postJsonWithFailover(
      '/register',
      {'username': username, 'email': email, 'password': password},
      timeout: const Duration(seconds: 120),
    );

    if (response.statusCode >= 400) {
      final errorMessage = _extractErrorMessage(response.body);
      throw Exception(errorMessage);
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<void> deleteRecord(int id) async {
    final urlsToTry = <String>[];
    if (_cachedWorkingBaseUrl != null) urlsToTry.add(_cachedWorkingBaseUrl!);
    for (final url in candidateUrls) {
      if (!urlsToTry.contains(url)) urlsToTry.add(url);
    }

    Object? lastError;
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base/records/$id');
        final response = await http.delete(uri, headers: {'Connection': 'close'}).timeout(const Duration(seconds: 120));

        if (_isServerInfrastructureError(response.statusCode, response.body)) {
          if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
          lastError = 'Host returned status ${response.statusCode}';
          continue;
        }

        if (response.statusCode >= 400) {
          final errorMessage = _extractErrorMessage(response.body);
          throw Exception(errorMessage);
        }
        _cachedWorkingBaseUrl = base;
        return;
      } catch (e) {
        if (_cachedWorkingBaseUrl == base) _cachedWorkingBaseUrl = null;
        lastError = e;
      }
    }
    throw Exception('Network error while contacting server: $lastError');
  }

  String _extractErrorMessage(String body) {
    try {
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      if (decoded['detail'] != null) {
        final detail = decoded['detail'];
        if (detail is List && detail.isNotEmpty) {
          final first = detail.first;
          if (first is Map && first['msg'] != null) {
            return first['msg'].toString();
          }
          return detail.toString();
        }
        return detail.toString();
      }
      if (decoded['message'] != null) return decoded['message'].toString();
      if (decoded['error'] != null) return decoded['error'].toString();
      return 'Operation failed.';
    } catch (_) {
      if (body.trim().isNotEmpty && body.length < 300) {
        return body.trim();
      }
      return 'Operation failed.';
    }
  }
}
