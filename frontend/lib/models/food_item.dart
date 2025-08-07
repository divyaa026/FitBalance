class FoodItem {
  final String name;
  final double proteinContent;
  final double calories;
  final double confidence;
  final List<String> tags;
  final Map<String, double> nutrients;

  FoodItem({
    required this.name,
    required this.proteinContent,
    required this.calories,
    required this.confidence,
    this.tags = const [],
    this.nutrients = const {},
  });

  factory FoodItem.fromJson(Map<String, dynamic> json) {
    return FoodItem(
      name: json['name'] ?? '',
      proteinContent: (json['protein_content'] ?? 0.0).toDouble(),
      calories: (json['calories'] ?? 0.0).toDouble(),
      confidence: (json['confidence'] ?? 0.0).toDouble(),
      tags: List<String>.from(json['tags'] ?? []),
      nutrients: Map<String, double>.from(json['nutrients'] ?? {}),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'protein_content': proteinContent,
      'calories': calories,
      'confidence': confidence,
      'tags': tags,
      'nutrients': nutrients,
    };
  }

  @override
  String toString() {
    return 'FoodItem(name: $name, protein: ${proteinContent}g, calories: $calories, confidence: $confidence)';
  }
} 