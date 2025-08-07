import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import '../models/food_item.dart';
import '../models/meal_analysis.dart';

class NutritionProvider with ChangeNotifier {
  double _dailyProtein = 0.0;
  double _dailyCalories = 0.0;
  List<MealAnalysis> _mealHistory = [];
  bool _isLoading = false;
  String? _error;

  // Getters
  double get dailyProtein => _dailyProtein;
  double get dailyCalories => _dailyCalories;
  List<MealAnalysis> get mealHistory => _mealHistory;
  bool get isLoading => _isLoading;
  String? get error => _error;

  // API base URL (should be configured based on environment)
  static const String _baseUrl = 'http://localhost:8000';

  /// Analyze a meal from an image file
  Future<MealAnalysis> analyzeMeal(String imagePath) async {
    _setLoading(true);
    _clearError();

    try {
      // Create multipart request
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/nutrition/analyze-meal'),
      );

      // Add image file
      final file = File(imagePath);
      final stream = http.ByteStream(file.openRead());
      final length = await file.length();
      
      final multipartFile = http.MultipartFile(
        'image',
        stream,
        length,
        filename: 'meal_image.jpg',
      );
      
      request.files.add(multipartFile);

      // Send request
      final response = await request.send();
      final responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final jsonData = json.decode(responseData);
        final analysis = MealAnalysis.fromJson(jsonData);
        
        // Update daily totals
        _updateDailyTotals(analysis);
        
        // Add to meal history
        _addToMealHistory(analysis);
        
        _setLoading(false);
        return analysis;
      } else {
        throw Exception('Failed to analyze meal: ${response.statusCode}');
      }
    } catch (e) {
      _setError('Error analyzing meal: $e');
      _setLoading(false);
      
      // Return mock data for development
      return _createMockAnalysis();
    }
  }

  /// Get nutrition recommendations for a user
  Future<List<String>> getRecommendations(String userId) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/nutrition/recommendations/$userId'),
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        return List<String>.from(jsonData['recommendations'] ?? []);
      } else {
        throw Exception('Failed to get recommendations: ${response.statusCode}');
      }
    } catch (e) {
      _setError('Error getting recommendations: $e');
      return _getMockRecommendations();
    }
  }

  /// Update daily protein target
  void updateDailyProteinTarget(double target) {
    // This would typically be saved to backend
    notifyListeners();
  }

  /// Reset daily totals
  void resetDailyTotals() {
    _dailyProtein = 0.0;
    _dailyCalories = 0.0;
    notifyListeners();
  }

  /// Clear meal history
  void clearMealHistory() {
    _mealHistory.clear();
    notifyListeners();
  }

  /// Get today's meals
  List<MealAnalysis> getTodayMeals() {
    final today = DateTime.now();
    return _mealHistory.where((meal) {
      return meal.timestamp.year == today.year &&
             meal.timestamp.month == today.month &&
             meal.timestamp.day == today.day;
    }).toList();
  }

  /// Get weekly nutrition summary
  Map<String, double> getWeeklySummary() {
    final weekAgo = DateTime.now().subtract(const Duration(days: 7));
    final weeklyMeals = _mealHistory.where((meal) {
      return meal.timestamp.isAfter(weekAgo);
    }).toList();

    double totalProtein = 0.0;
    double totalCalories = 0.0;
    double totalCarbs = 0.0;
    double totalFat = 0.0;
    double totalFiber = 0.0;

    for (final meal in weeklyMeals) {
      totalProtein += meal.totalProtein;
      totalCalories += meal.totalCalories;
      totalCarbs += meal.totalCarbs;
      totalFat += meal.totalFat;
      totalFiber += meal.totalFiber;
    }

    return {
      'protein': totalProtein,
      'calories': totalCalories,
      'carbs': totalCarbs,
      'fat': totalFat,
      'fiber': totalFiber,
    };
  }

  // Private methods

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

  void _updateDailyTotals(MealAnalysis analysis) {
    _dailyProtein += analysis.totalProtein;
    _dailyCalories += analysis.totalCalories;
    notifyListeners();
  }

  void _addToMealHistory(MealAnalysis analysis) {
    _mealHistory.add(analysis);
    notifyListeners();
  }

  // Mock data for development
  MealAnalysis _createMockAnalysis() {
    final mockFoods = [
      FoodItem(
        name: 'grilled_chicken',
        proteinContent: 25.0,
        calories: 165.0,
        confidence: 0.92,
        tags: ['protein', 'lean', 'healthy'],
        nutrients: {
          'protein': 25.0,
          'fat': 3.6,
          'carbs': 0.0,
          'fiber': 0.0,
        },
      ),
      FoodItem(
        name: 'brown_rice',
        proteinContent: 4.5,
        calories: 216.0,
        confidence: 0.88,
        tags: ['grain', 'fiber', 'complex_carbs'],
        nutrients: {
          'protein': 4.5,
          'fat': 1.8,
          'carbs': 45.0,
          'fiber': 3.5,
        },
      ),
      FoodItem(
        name: 'broccoli',
        proteinContent: 2.8,
        calories: 55.0,
        confidence: 0.95,
        tags: ['vegetable', 'fiber', 'vitamins'],
        nutrients: {
          'protein': 2.8,
          'fat': 0.6,
          'carbs': 11.0,
          'fiber': 5.2,
        },
      ),
    ];

    return MealAnalysis(
      detectedFoods: mockFoods,
      totalProtein: 32.3,
      totalCalories: 436.0,
      totalCarbs: 56.0,
      totalFat: 6.0,
      totalFiber: 8.7,
      mealQualityScore: 85.0,
      recommendations: [
        'Great protein content! This meal provides excellent muscle-building nutrients.',
        'Consider adding a healthy fat source like avocado or nuts for better satiety.',
        'The fiber content is good, but you could add more vegetables for micronutrients.',
      ],
      shapExplanation: 'Reduced protein recommendation due to poor sleep quality (HRV=42ms) and high stress levels detected from your recent activity patterns.',
    );
  }

  List<String> _getMockRecommendations() {
    return [
      'Aim for 1.6-2.2g of protein per kg of body weight for optimal muscle growth.',
      'Spread protein intake evenly across 3-4 meals throughout the day.',
      'Include a variety of protein sources: lean meats, fish, eggs, dairy, and plant-based options.',
      'Consider your activity level when adjusting protein intake.',
      'Monitor your recovery and adjust based on how you feel.',
    ];
  }
} 