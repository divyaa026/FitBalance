import 'package:flutter/material.dart';
import '../models/food_item.dart';

class FoodDetectionOverlay extends StatelessWidget {
  final List<FoodItem> foods;

  const FoodDetectionOverlay({
    Key? key,
    required this.foods,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: foods.map((food) => _buildFoodBoundingBox(food)).toList(),
    );
  }

  Widget _buildFoodBoundingBox(FoodItem food) {
    // Mock bounding box coordinates (in real implementation, these would come from the ML model)
    final boundingBox = food.boundingBox;
    
    return Positioned(
      left: boundingBox.left,
      top: boundingBox.top,
      child: Container(
        width: boundingBox.width,
        height: boundingBox.height,
        decoration: BoxDecoration(
          border: Border.all(
            color: _getConfidenceColor(food.confidence),
            width: 2,
          ),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Stack(
          children: [
            // Food label
            Positioned(
              top: -25,
              left: 0,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: _getConfidenceColor(food.confidence),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '${food.name.replaceAll('_', ' ').toUpperCase()} (${(food.confidence * 100).toStringAsFixed(0)}%)',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
            
            // Protein content indicator
            Positioned(
              bottom: -20,
              right: 0,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: Colors.blue[600],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '${food.proteinContent}g protein',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 8,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) {
      return Colors.green;
    } else if (confidence >= 0.6) {
      return Colors.orange;
    } else {
      return Colors.red;
    }
  }
}

// Extension to add bounding box to FoodItem
extension FoodItemBoundingBox on FoodItem {
  Rect get boundingBox {
    // Mock bounding box - in real implementation, this would come from the ML model
    return Rect.fromLTWH(
      (hashCode % 300).toDouble(), // Mock left position
      (hashCode % 200).toDouble(), // Mock top position
      100, // Mock width
      80,  // Mock height
    );
  }
} 