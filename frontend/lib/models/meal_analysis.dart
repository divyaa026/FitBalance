import 'food_item.dart';

class MealAnalysis {
  final List<FoodItem> detectedFoods;
  final double totalProtein;
  final double totalCalories;
  final double totalCarbs;
  final double totalFat;
  final double totalFiber;
  final double mealQualityScore;
  final List<String> recommendations;
  final String? shapExplanation;
  final DateTime timestamp;

  MealAnalysis({
    required this.detectedFoods,
    required this.totalProtein,
    required this.totalCalories,
    required this.totalCarbs,
    required this.totalFat,
    required this.totalFiber,
    required this.mealQualityScore,
    required this.recommendations,
    this.shapExplanation,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  factory MealAnalysis.fromJson(Map<String, dynamic> json) {
    return MealAnalysis(
      detectedFoods: (json['detected_foods'] as List?)
          ?.map((food) => FoodItem.fromJson(food))
          .toList() ?? [],
      totalProtein: (json['total_protein'] ?? 0.0).toDouble(),
      totalCalories: (json['total_calories'] ?? 0.0).toDouble(),
      totalCarbs: (json['total_carbs'] ?? 0.0).toDouble(),
      totalFat: (json['total_fat'] ?? 0.0).toDouble(),
      totalFiber: (json['total_fiber'] ?? 0.0).toDouble(),
      mealQualityScore: (json['meal_quality_score'] ?? 0.0).toDouble(),
      recommendations: List<String>.from(json['recommendations'] ?? []),
      shapExplanation: json['shap_explanation'],
      timestamp: json['timestamp'] != null 
          ? DateTime.parse(json['timestamp']) 
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'detected_foods': detectedFoods.map((food) => food.toJson()).toList(),
      'total_protein': totalProtein,
      'total_calories': totalCalories,
      'total_carbs': totalCarbs,
      'total_fat': totalFat,
      'total_fiber': totalFiber,
      'meal_quality_score': mealQualityScore,
      'recommendations': recommendations,
      'shap_explanation': shapExplanation,
      'timestamp': timestamp.toIso8601String(),
    };
  }

  @override
  String toString() {
    return 'MealAnalysis(protein: ${totalProtein}g, calories: $totalCalories, quality: $mealQualityScore)';
  }
} 