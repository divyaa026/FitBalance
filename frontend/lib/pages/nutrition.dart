import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:provider/provider.dart';
import '../providers/nutrition_provider.dart';
import '../widgets/camera_preview.dart';
import '../widgets/protein_gauge.dart';
import '../widgets/shap_explanation_bubble.dart';
import '../widgets/food_detection_overlay.dart';
import '../widgets/nutrition_recommendations.dart';
import '../models/food_item.dart';
import '../models/meal_analysis.dart';

class NutritionPage extends StatefulWidget {
  const NutritionPage({Key? key}) : super(key: key);

  @override
  _NutritionPageState createState() => _NutritionPageState();
}

class _NutritionPageState extends State<NutritionPage> {
  CameraController? _cameraController;
  bool _isAnalyzing = false;
  bool _showCamera = true;
  List<FoodItem> _detectedFoods = [];
  MealAnalysis? _currentAnalysis;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    if (cameras.isNotEmpty) {
      _cameraController = CameraController(
        cameras[0],
        ResolutionPreset.high,
        enableAudio: false,
      );
      await _cameraController!.initialize();
      if (mounted) {
        setState(() {});
      }
    }
  }

  Future<void> _captureAndAnalyze() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    setState(() {
      _isAnalyzing = true;
    });

    try {
      // Capture image
      final image = await _cameraController!.takePicture();
      
      // Analyze meal
      final nutritionProvider = context.read<NutritionProvider>();
      final analysis = await nutritionProvider.analyzeMeal(image.path);
      
      setState(() {
        _currentAnalysis = analysis;
        _detectedFoods = analysis.detectedFoods;
        _showCamera = false;
        _isAnalyzing = false;
      });

      // Show results
      _showAnalysisResults(analysis);
      
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error analyzing meal: $e')),
      );
    }
  }

  void _showAnalysisResults(MealAnalysis analysis) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        builder: (context, scrollController) => Container(
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            children: [
              // Handle
              Container(
                margin: const EdgeInsets.symmetric(vertical: 8),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Expanded(
                child: SingleChildScrollView(
                  controller: scrollController,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Header
                        Row(
                          children: [
                            Icon(Icons.restaurant, color: Colors.green[600]),
                            const SizedBox(width: 8),
                            Text(
                              'Meal Analysis Results',
                              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        
                        // Nutrition Summary
                        _buildNutritionSummary(analysis),
                        const SizedBox(height: 16),
                        
                        // Detected Foods
                        _buildDetectedFoods(analysis.detectedFoods),
                        const SizedBox(height: 16),
                        
                        // Recommendations
                        _buildRecommendations(analysis.recommendations),
                        const SizedBox(height: 16),
                        
                        // SHAP Explanation
                        if (analysis.shapExplanation != null)
                          _buildShapExplanation(analysis.shapExplanation!),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildNutritionSummary(MealAnalysis analysis) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Nutrition Summary',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildNutritionMetric('Protein', '${analysis.totalProtein}g', Colors.blue),
                _buildNutritionMetric('Calories', '${analysis.totalCalories}', Colors.orange),
                _buildNutritionMetric('Quality', '${analysis.mealQualityScore}/100', Colors.green),
              ],
            ),
            const SizedBox(height: 12),
            LinearProgressIndicator(
              value: analysis.totalProtein / 120, // Assuming 120g daily target
              backgroundColor: Colors.grey[200],
              valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
            ),
            const SizedBox(height: 4),
            Text(
              'Daily Protein Progress: ${(analysis.totalProtein / 120 * 100).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildNutritionMetric(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            color: color,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }

  Widget _buildDetectedFoods(List<FoodItem> foods) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Detected Foods',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            ...foods.map((food) => _buildFoodItem(food)).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildFoodItem(FoodItem food) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: Colors.grey[200],
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(Icons.restaurant, color: Colors.grey[600]),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  food.name.replaceAll('_', ' ').toUpperCase(),
                  style: const TextStyle(fontWeight: FontWeight.w500),
                ),
                Text(
                  '${food.proteinContent}g protein • ${food.calories} cal',
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.green[100],
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              '${(food.confidence * 100).toStringAsFixed(0)}%',
              style: TextStyle(
                color: Colors.green[700],
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendations(List<String> recommendations) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.lightbulb, color: Colors.amber[600]),
                const SizedBox(width: 8),
                Text(
                  'Recommendations',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            ...recommendations.map((rec) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('• ', style: TextStyle(fontSize: 16)),
                  Expanded(child: Text(rec)),
                ],
              ),
            )).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildShapExplanation(String explanation) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.psychology, color: Colors.purple[600]),
                const SizedBox(width: 8),
                Text(
                  'AI Explanation',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.purple[50],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.purple[200]!),
              ),
              child: Text(
                explanation,
                style: TextStyle(
                  color: Colors.purple[800],
                  fontStyle: FontStyle.italic,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.green[50],
              border: Border(bottom: BorderSide(color: Colors.green[200]!)),
            ),
            child: Row(
              children: [
                Icon(Icons.restaurant, color: Colors.green[600], size: 28),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Nutrition Analysis',
                        style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Colors.green[800],
                        ),
                      ),
                      Text(
                        'AI-powered meal analysis and recommendations',
                        style: TextStyle(
                          color: Colors.green[600],
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  onPressed: () {
                    setState(() {
                      _showCamera = !_showCamera;
                    });
                  },
                  icon: Icon(
                    _showCamera ? Icons.photo_library : Icons.camera_alt,
                    color: Colors.green[600],
                  ),
                ),
              ],
            ),
          ),
          
          // Main Content
          Expanded(
            child: _showCamera ? _buildCameraView() : _buildAnalysisView(),
          ),
          
          // Bottom Controls
          _buildBottomControls(),
        ],
      ),
    );
  }

  Widget _buildCameraView() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    return Stack(
      children: [
        // Camera Preview
        CameraPreview(_cameraController!),
        
        // Food Detection Overlay
        if (_detectedFoods.isNotEmpty)
          FoodDetectionOverlay(foods: _detectedFoods),
        
        // Camera Controls
        Positioned(
          bottom: 20,
          left: 0,
          right: 0,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              // Gallery Button
              FloatingActionButton(
                onPressed: () {
                  // TODO: Implement gallery picker
                },
                backgroundColor: Colors.white,
                child: Icon(Icons.photo_library, color: Colors.green[600]),
              ),
              
              // Capture Button
              FloatingActionButton(
                onPressed: _isAnalyzing ? null : _captureAndAnalyze,
                backgroundColor: Colors.green[600],
                child: _isAnalyzing
                    ? const CircularProgressIndicator(color: Colors.white)
                    : const Icon(Icons.camera_alt, color: Colors.white),
              ),
              
              // Settings Button
              FloatingActionButton(
                onPressed: () {
                  // TODO: Implement camera settings
                },
                backgroundColor: Colors.white,
                child: Icon(Icons.settings, color: Colors.green[600]),
              ),
            ],
          ),
        ),
        
        // Analysis Status
        if (_isAnalyzing)
          Positioned(
            top: 20,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Analyzing meal...',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildAnalysisView() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Protein Gauge
          ProteinGauge(
            protein: context.watch<NutritionProvider>().dailyProtein,
            targetProtein: 120,
          ),
          const SizedBox(height: 20),
          
          // SHAP Explanation Bubble
          if (_currentAnalysis?.shapExplanation != null)
            ShapExplanationBubble(
              message: _currentAnalysis!.shapExplanation!,
            ),
          const SizedBox(height: 20),
          
          // Nutrition Recommendations
          if (_currentAnalysis != null)
            NutritionRecommendations(analysis: _currentAnalysis!),
          
          // Retake Photo Button
          ElevatedButton.icon(
            onPressed: () {
              setState(() {
                _showCamera = true;
                _detectedFoods.clear();
                _currentAnalysis = null;
              });
            },
            icon: const Icon(Icons.camera_alt),
            label: const Text('Take New Photo'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green[600],
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomControls() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: Colors.grey[300]!)),
      ),
      child: Row(
        children: [
          // Daily Summary
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Today\'s Protein',
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 12,
                  ),
                ),
                Text(
                  '${context.watch<NutritionProvider>().dailyProtein}g / 120g',
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
          ),
          
          // Progress Indicator
          Expanded(
            child: Column(
              children: [
                LinearProgressIndicator(
                  value: context.watch<NutritionProvider>().dailyProtein / 120,
                  backgroundColor: Colors.grey[200],
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.green[600]!),
                ),
                const SizedBox(height: 4),
                Text(
                  '${(context.watch<NutritionProvider>().dailyProtein / 120 * 100).toStringAsFixed(0)}%',
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          
          // Quick Actions
          Row(
            children: [
              IconButton(
                onPressed: () {
                  // TODO: Show meal history
                },
                icon: Icon(Icons.history, color: Colors.green[600]),
              ),
              IconButton(
                onPressed: () {
                  // TODO: Show nutrition insights
                },
                icon: Icon(Icons.insights, color: Colors.green[600]),
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }
} 