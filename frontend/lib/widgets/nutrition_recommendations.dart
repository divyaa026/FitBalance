import 'package:flutter/material.dart';
import '../models/meal_analysis.dart';

class NutritionRecommendations extends StatelessWidget {
  final MealAnalysis analysis;

  const NutritionRecommendations({
    Key? key,
    required this.analysis,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Icon(Icons.lightbulb, color: Colors.amber[600]),
                const SizedBox(width: 8),
                Text(
                  'Personalized Recommendations',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            
            // Meal Quality Score
            _buildQualityScore(),
            const SizedBox(height: 16),
            
            // Recommendations List
            ...analysis.recommendations.map((rec) => _buildRecommendationItem(rec)).toList(),
            
            const SizedBox(height: 16),
            
            // Action Buttons
            _buildActionButtons(context),
          ],
        ),
      ),
    );
  }

  Widget _buildQualityScore() {
    final score = analysis.mealQualityScore;
    Color scoreColor;
    String scoreLabel;
    
    if (score >= 80) {
      scoreColor = Colors.green;
      scoreLabel = 'Excellent';
    } else if (score >= 60) {
      scoreColor = Colors.orange;
      scoreLabel = 'Good';
    } else {
      scoreColor = Colors.red;
      scoreLabel = 'Needs Improvement';
    }
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: scoreColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: scoreColor.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Icon(Icons.star, color: scoreColor),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Meal Quality Score',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: scoreColor,
                  ),
                ),
                Text(
                  '$scoreLabel ($score/100)',
                  style: TextStyle(
                    color: scoreColor,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          Text(
            '$score',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: scoreColor,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendationItem(String recommendation) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            margin: const EdgeInsets.only(top: 6),
            width: 8,
            height: 8,
            decoration: const BoxDecoration(
              color: Colors.blue,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              recommendation,
              style: const TextStyle(
                fontSize: 14,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: OutlinedButton.icon(
            onPressed: () {
              _showDetailedAnalysis(context);
            },
            icon: const Icon(Icons.analytics),
            label: const Text('Detailed Analysis'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.blue[600],
              side: BorderSide(color: Colors.blue[600]!),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () {
              _saveMeal(context);
            },
            icon: const Icon(Icons.save),
            label: const Text('Save Meal'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green[600],
              foregroundColor: Colors.white,
            ),
          ),
        ),
      ],
    );
  }

  void _showDetailedAnalysis(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Detailed Nutrition Analysis'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildNutritionDetail('Total Protein', '${analysis.totalProtein}g'),
              _buildNutritionDetail('Total Calories', '${analysis.totalCalories}'),
              _buildNutritionDetail('Carbohydrates', '${analysis.totalCarbs}g'),
              _buildNutritionDetail('Fat', '${analysis.totalFat}g'),
              _buildNutritionDetail('Fiber', '${analysis.totalFiber}g'),
              const SizedBox(height: 16),
              const Text(
                'Macronutrient Balance:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              _buildMacroBar('Protein', analysis.totalProtein, analysis.totalCalories * 0.3),
              _buildMacroBar('Carbs', analysis.totalCarbs * 4, analysis.totalCalories * 0.4),
              _buildMacroBar('Fat', analysis.totalFat * 9, analysis.totalCalories * 0.3),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _buildNutritionDetail(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildMacroBar(String macro, double actual, double target) {
    final percentage = (actual / target * 100).clamp(0.0, 100.0);
    Color color;
    
    if (percentage >= 80 && percentage <= 120) {
      color = Colors.green;
    } else if (percentage >= 60 && percentage <= 140) {
      color = Colors.orange;
    } else {
      color = Colors.red;
    }
    
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(macro),
              Text('${percentage.toStringAsFixed(0)}%'),
            ],
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(
            value: percentage / 100,
            backgroundColor: Colors.grey[200],
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ],
      ),
    );
  }

  void _saveMeal(BuildContext context) {
    // TODO: Implement meal saving functionality
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Meal saved to your nutrition history!'),
        backgroundColor: Colors.green,
      ),
    );
  }
} 