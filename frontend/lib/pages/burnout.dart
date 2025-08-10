import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/burnout_provider.dart';

class BurnoutPage extends StatefulWidget {
  const BurnoutPage({Key? key}) : super(key: key);

  @override
  _BurnoutPageState createState() => _BurnoutPageState();
}

class _BurnoutPageState extends State<BurnoutPage> {
  final _formKey = GlobalKey<FormState>();
  final _workoutFrequencyController = TextEditingController(text: '5');
  final _sleepHoursController = TextEditingController(text: '7.5');
  final _stressLevelController = TextEditingController(text: '5');
  final _recoveryTimeController = TextEditingController(text: '2');
  String _performanceTrend = 'stable';

  Map<String, dynamic>? _lastRiskAnalysis;
  Map<String, dynamic>? _lastSurvivalCurve;
  List<String>? _lastRecommendations;

  final List<String> _performanceTrends = ['improving', 'stable', 'declining'];

  @override
  void dispose() {
    _workoutFrequencyController.dispose();
    _sleepHoursController.dispose();
    _stressLevelController.dispose();
    _recoveryTimeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Center(
              child: Column(
                children: [
                  Icon(Icons.psychology, size: 80, color: Colors.orange),
                  SizedBox(height: 16),
                  Text(
                    'Burnout Risk Assessment',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'AI-powered burnout prediction',
                    style: TextStyle(fontSize: 16, color: Colors.grey[600]),
                  ),
                ],
              ),
            ),
            SizedBox(height: 32),

            // Assessment Form
            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Enter Your Metrics:',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 16),

                      // Workout Frequency
                      TextFormField(
                        controller: _workoutFrequencyController,
                        decoration: InputDecoration(
                          labelText: 'Workouts per week',
                          border: OutlineInputBorder(),
                          helperText: 'How many times do you work out per week?',
                        ),
                        keyboardType: TextInputType.number,
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter workout frequency';
                          }
                          final num = int.tryParse(value);
                          if (num == null || num < 0 || num > 7) {
                            return 'Please enter a valid number (0-7)';
                          }
                          return null;
                        },
                      ),
                      SizedBox(height: 16),

                      // Sleep Hours
                      TextFormField(
                        controller: _sleepHoursController,
                        decoration: InputDecoration(
                          labelText: 'Sleep hours per night',
                          border: OutlineInputBorder(),
                          helperText: 'Average hours of sleep per night',
                        ),
                        keyboardType: TextInputType.number,
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter sleep hours';
                          }
                          final num = double.tryParse(value);
                          if (num == null || num < 0 || num > 24) {
                            return 'Please enter a valid number (0-24)';
                          }
                          return null;
                        },
                      ),
                      SizedBox(height: 16),

                      // Stress Level
                      TextFormField(
                        controller: _stressLevelController,
                        decoration: InputDecoration(
                          labelText: 'Stress level (1-10)',
                          border: OutlineInputBorder(),
                          helperText: 'Rate your current stress level from 1 (low) to 10 (high)',
                        ),
                        keyboardType: TextInputType.number,
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter stress level';
                          }
                          final num = int.tryParse(value);
                          if (num == null || num < 1 || num > 10) {
                            return 'Please enter a valid number (1-10)';
                          }
                          return null;
                        },
                      ),
                      SizedBox(height: 16),

                      // Recovery Time
                      TextFormField(
                        controller: _recoveryTimeController,
                        decoration: InputDecoration(
                          labelText: 'Recovery time (days)',
                          border: OutlineInputBorder(),
                          helperText: 'How many days do you need to recover between intense sessions?',
                        ),
                        keyboardType: TextInputType.number,
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter recovery time';
                          }
                          final num = int.tryParse(value);
                          if (num == null || num < 0 || num > 7) {
                            return 'Please enter a valid number (0-7)';
                          }
                          return null;
                        },
                      ),
                      SizedBox(height: 16),

                      // Performance Trend
                      DropdownButtonFormField<String>(
                        value: _performanceTrend,
                        decoration: InputDecoration(
                          labelText: 'Performance trend',
                          border: OutlineInputBorder(),
                          helperText: 'How is your performance trending?',
                        ),
                        items: _performanceTrends.map((trend) {
                          return DropdownMenuItem(
                            value: trend,
                            child: Text(trend.toUpperCase()),
                          );
                        }).toList(),
                        onChanged: (value) {
                          setState(() {
                            _performanceTrend = value!;
                          });
                        },
                      ),
                      SizedBox(height: 24),

                      // Submit Button
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton.icon(
                          onPressed: _assessRisk,
                          icon: Icon(Icons.assessment),
                          label: Text('Assess Burnout Risk'),
                          style: ElevatedButton.styleFrom(
                            padding: EdgeInsets.symmetric(vertical: 16),
                            backgroundColor: Colors.orange,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            SizedBox(height: 24),

            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _getSurvivalCurve,
                    icon: Icon(Icons.timeline),
                    label: Text('Survival Curve'),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _getRecommendations,
                    icon: Icon(Icons.lightbulb),
                    label: Text('Recommendations'),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 32),

            // Loading Indicator
            Consumer<BurnoutProvider>(
              builder: (context, provider, child) {
                if (provider.isLoading) {
                  return Center(
                    child: Column(
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text('Analyzing burnout risk...'),
                      ],
                    ),
                  );
                }
                return SizedBox.shrink();
              },
            ),

            // Error Display
            Consumer<BurnoutProvider>(
              builder: (context, provider, child) {
                if (provider.error != null) {
                  return Container(
                    padding: EdgeInsets.all(16),
                    margin: EdgeInsets.only(bottom: 16),
                    decoration: BoxDecoration(
                      color: Colors.red[50],
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.red[200]!),
                    ),
                    child: Text(
                      provider.error!,
                      style: TextStyle(color: Colors.red[700]),
                    ),
                  );
                }
                return SizedBox.shrink();
              },
            ),

            // Risk Analysis Results
            if (_lastRiskAnalysis != null) ...[
              Text(
                'Risk Analysis:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 16),
              _buildRiskAnalysisCard(_lastRiskAnalysis!),
            ],

            // Survival Curve Results
            if (_lastSurvivalCurve != null) ...[
              SizedBox(height: 24),
              Text(
                'Survival Curve:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 16),
              _buildSurvivalCurveCard(_lastSurvivalCurve!),
            ],

            // Recommendations Results
            if (_lastRecommendations != null) ...[
              SizedBox(height: 24),
              Text(
                'Recommendations:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 16),
              _buildRecommendationsCard(_lastRecommendations!),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _assessRisk() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    final provider = context.read<BurnoutProvider>();
    final analysis = await provider.analyzeBurnoutRisk(
      userId: 'default_user',
      workoutFrequency: int.parse(_workoutFrequencyController.text),
      sleepHours: double.parse(_sleepHoursController.text),
      stressLevel: int.parse(_stressLevelController.text),
      recoveryTime: int.parse(_recoveryTimeController.text),
      performanceTrend: _performanceTrend,
    );

    setState(() {
      _lastRiskAnalysis = analysis;
    });

    _showRiskResults(analysis);
  }

  Future<void> _getSurvivalCurve() async {
    final provider = context.read<BurnoutProvider>();
    final curve = await provider.getSurvivalCurve('default_user');
    
    setState(() {
      _lastSurvivalCurve = curve;
    });
  }

  Future<void> _getRecommendations() async {
    final provider = context.read<BurnoutProvider>();
    final recommendations = await provider.getBurnoutRecommendations('default_user');
    
    setState(() {
      _lastRecommendations = recommendations;
    });
  }

  void _showRiskResults(Map<String, dynamic> analysis) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        builder: (context, scrollController) => Container(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            children: [
              Container(
                margin: EdgeInsets.only(top: 8),
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
                  padding: EdgeInsets.all(16),
                  child: _buildRiskAnalysisCard(analysis),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRiskAnalysisCard(Map<String, dynamic> analysis) {
    final riskLevel = analysis['risk_level'] ?? 'unknown';
    final riskScore = analysis['risk_score'] ?? 0;
    final factors = analysis['factors'] ?? {};
    final predictions = analysis['predictions'] ?? {};

    Color riskColor;
    switch (riskLevel) {
      case 'low':
        riskColor = Colors.green;
        break;
      case 'medium':
        riskColor = Colors.orange;
        break;
      case 'high':
        riskColor = Colors.red;
        break;
      default:
        riskColor = Colors.grey;
    }

    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Risk Level and Score
            Row(
              children: [
                Icon(Icons.warning, color: riskColor),
                SizedBox(width: 8),
                Text(
                  'Risk Level: ${riskLevel.toUpperCase()}',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: riskColor),
                ),
                Spacer(),
                Text(
                  'Score: $riskScore%',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            SizedBox(height: 16),

            // Risk Factors
            Text(
              'Risk Factors:',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            ...factors.entries.map((entry) => Padding(
              padding: EdgeInsets.only(bottom: 4),
              child: Row(
                children: [
                  Icon(Icons.info, color: Colors.blue, size: 16),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text('${entry.key.replaceAll('_', ' ').toUpperCase()}: ${entry.value}'),
                  ),
                ],
              ),
            )),
            SizedBox(height: 16),

            // Predictions
            Text(
              'Predictions:',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            ...predictions.entries.map((entry) => Padding(
              padding: EdgeInsets.only(bottom: 4),
              child: Row(
                children: [
                  Icon(Icons.timeline, color: Colors.purple, size: 16),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text('${entry.key.replaceAll('_', ' ').toUpperCase()}: ${entry.value}'),
                  ),
                ],
              ),
            )),
          ],
        ),
      ),
    );
  }

  Widget _buildSurvivalCurveCard(Map<String, dynamic> curve) {
    final survivalData = curve['survival_data'] ?? [];
    final medianSurvival = curve['median_survival_days'] ?? 0;

    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Survival Curve Data',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text('Median Survival: $medianSurvival days'),
            SizedBox(height: 16),
            Container(
              height: 200,
              child: Center(
                child: Text('Survival curve chart would appear here'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecommendationsCard(List<String> recommendations) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Personalized Recommendations:',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            ...recommendations.map((rec) => Padding(
              padding: EdgeInsets.only(bottom: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.lightbulb, color: Colors.green, size: 16),
                  SizedBox(width: 8),
                  Expanded(child: Text(rec)),
                ],
              ),
            )),
          ],
        ),
      ),
    );
  }
}
