import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

class BiomechanicsProvider with ChangeNotifier {
  bool _isLoading = false;
  String? _error;
  Map<String, dynamic>? _lastAnalysis;
  String? _lastHeatmap;

  // Getters
  bool get isLoading => _isLoading;
  String? get error => _error;
  Map<String, dynamic>? get lastAnalysis => _lastAnalysis;
  String? get lastHeatmap => _lastHeatmap;

  // API base URL
  static const String _baseUrl = 'http://localhost:8000';

  /// Analyze biomechanics from video file
  Future<Map<String, dynamic>> analyzeMovement(String videoPath, String exerciseType, String userId) async {
    _setLoading(true);
    _clearError();

    try {
      // Create multipart request
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/biomechanics/analyze'),
      );

      // Add video file
      final file = File(videoPath);
      final stream = http.ByteStream(file.openRead());
      final length = await file.length();
      
      final multipartFile = http.MultipartFile(
        'video_file',
        stream,
        length,
        filename: 'exercise_video.mp4',
      );
      
      request.files.add(multipartFile);

      // Add form fields
      request.fields['exercise_type'] = exerciseType;
      request.fields['user_id'] = userId;

      // Send request
      final response = await request.send();
      final responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final jsonData = json.decode(responseData);
        _lastAnalysis = jsonData;
        _setLoading(false);
        return jsonData;
      } else {
        throw Exception('Failed to analyze movement: ${response.statusCode}');
      }
    } catch (e) {
      _setError('Error analyzing movement: $e');
      _setLoading(false);
      
      // Return mock data for development
      return _createMockAnalysis();
    }
  }

  /// Get torque heatmap for user's exercise
  Future<String> getTorqueHeatmap(String userId, String exerciseType) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/biomechanics/heatmap/$userId?exercise_type=$exerciseType'),
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        _lastHeatmap = jsonData['heatmap'];
        return jsonData['heatmap'];
      } else {
        throw Exception('Failed to get heatmap: ${response.statusCode}');
      }
    } catch (e) {
      _setError('Error getting heatmap: $e');
      return _getMockHeatmap();
    }
  }

  // Private helper methods
  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void _setError(String error) {
    _error = error;
    notifyListeners();
  }

  void _clearError() {
    _error = null;
    notifyListeners();
  }

  // Mock data for development
  Map<String, dynamic> _createMockAnalysis() {
    return {
      'exercise_type': 'squat',
      'user_id': 'default',
      'analysis': {
        'form_score': 85,
        'issues': [
          'Knees slightly caving in',
          'Depth could be deeper'
        ],
        'recommendations': [
          'Focus on pushing knees out',
          'Aim for thighs parallel to ground'
        ],
        'joint_angles': {
          'knee': [120, 115, 110],
          'hip': [90, 85, 80],
          'ankle': [15, 20, 25]
        }
      },
      'timestamp': DateTime.now().toIso8601String()
    };
  }

  String _getMockHeatmap() {
    return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
  }
}
