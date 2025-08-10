import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class BurnoutProvider with ChangeNotifier {
  bool _isLoading = false;
  String? _error;
  Map<String, dynamic>? _lastRiskAnalysis;
  Map<String, dynamic>? _lastSurvivalCurve;
  List<String>? _lastRecommendations;

  // Getters
  bool get isLoading => _isLoading;
  String? get error => _error;
  Map<String, dynamic>? get lastRiskAnalysis => _lastRiskAnalysis;
  Map<String, dynamic>? get lastSurvivalCurve => _lastSurvivalCurve;
  List<String>? get lastRecommendations => _lastRecommendations;

  // API base URL
  static const String _baseUrl = 'http://localhost:8000';

  /// Analyze burnout risk based on user metrics
  Future<Map<String, dynamic>> analyzeBurnoutRisk({
    required String userId,
    required int workoutFrequency,
    required double sleepHours,
    required int stressLevel,
    required int recoveryTime,
    String performanceTrend = 'stable',
  }) async {
    _setLoading(true);
    _clearError();

    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/burnout/analyze'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'user_id': userId,
          'workout_frequency': workoutFrequency,
          'sleep_hours': sleepHours,
          'stress_level': stressLevel,
          'recovery_time': recoveryTime,
          'performance_trend': performanceTrend,
        }),
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        _lastRiskAnalysis = jsonData;
        _setLoading(false);
        return jsonData;
      } else {
        throw Exception('Failed to analyze burnout risk: ${response.statusCode}');
      }
    } catch (e) {
      _setError('Error analyzing burnout risk: $e');
      _setLoading(false);
      
      // Return mock data for development
      return _createMockRiskAnalysis();
    }
  }

  /// Get survival curve for burnout prediction
  Future<Map<String, dynamic>> getSurvivalCurve(String userId) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/burnout/survival-curve/$userId'),
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        _lastSurvivalCurve = jsonData;
        return jsonData;
      } else {
        throw Exception('Failed to get survival curve: ${response.statusCode}');
      }
    } catch (e) {
      _setError('Error getting survival curve: $e');
      return _createMockSurvivalCurve();
    }
  }

  /// Get personalized recommendations to prevent burnout
  Future<List<String>> getBurnoutRecommendations(String userId) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/burnout/recommendations/$userId'),
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        final recommendations = List<String>.from(jsonData['recommendations'] ?? []);
        _lastRecommendations = recommendations;
        return recommendations;
      } else {
        throw Exception('Failed to get recommendations: ${response.statusCode}');
      }
    } catch (e) {
      _setError('Error getting recommendations: $e');
      return _getMockRecommendations();
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
  Map<String, dynamic> _createMockRiskAnalysis() {
    return {
      'user_id': 'default',
      'risk_level': 'medium',
      'risk_score': 65,
      'factors': {
        'workout_frequency': 'High risk (6+ times/week)',
        'sleep_hours': 'Moderate risk (6.5 hours)',
        'stress_level': 'High risk (8/10)',
        'recovery_time': 'Low risk (2 days)',
        'performance_trend': 'Stable'
      },
      'predictions': {
        'burnout_probability_30_days': 0.25,
        'burnout_probability_90_days': 0.45,
        'recommended_rest_days': 2
      },
      'timestamp': DateTime.now().toIso8601String()
    };
  }

  Map<String, dynamic> _createMockSurvivalCurve() {
    return {
      'user_id': 'default',
      'survival_data': [
        {'days': 0, 'survival_probability': 1.0},
        {'days': 30, 'survival_probability': 0.85},
        {'days': 60, 'survival_probability': 0.70},
        {'days': 90, 'survival_probability': 0.55},
        {'days': 120, 'survival_probability': 0.40},
        {'days': 150, 'survival_probability': 0.25},
        {'days': 180, 'survival_probability': 0.15},
      ],
      'median_survival_days': 120,
      'timestamp': DateTime.now().toIso8601String()
    };
  }

  List<String> _getMockRecommendations() {
    return [
      'Reduce workout frequency to 4-5 times per week',
      'Increase sleep duration to 7-8 hours per night',
      'Implement stress management techniques (meditation, yoga)',
      'Take 2-3 rest days between intense training sessions',
      'Consider working with a coach to optimize training load',
      'Monitor heart rate variability for recovery assessment'
    ];
  }
}
