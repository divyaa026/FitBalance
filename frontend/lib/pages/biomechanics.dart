import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import '../providers/biomechanics_provider.dart';
import '../widgets/video_recorder.dart';

class BiomechanicsPage extends StatefulWidget {
  const BiomechanicsPage({Key? key}) : super(key: key);

  @override
  _BiomechanicsPageState createState() => _BiomechanicsPageState();
}

class _BiomechanicsPageState extends State<BiomechanicsPage> {
  final ImagePicker _picker = ImagePicker();
  String _selectedExercise = 'squat';
  Map<String, dynamic>? _lastAnalysis;
  String? _lastHeatmap;
  String? _lastVideoPath;
  bool _isAnalyzing = false;

  final List<String> _exerciseTypes = [
    'squat',
    'deadlift',
    'bench_press',
    'overhead_press',
    'row',
    'lunge'
  ];

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
                  Icon(Icons.fitness_center, size: 80, color: Colors.blue),
                  SizedBox(height: 16),
                  Text(
                    'Biomechanics Analysis',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'AI-powered movement analysis',
                    style: TextStyle(fontSize: 16, color: Colors.grey[600]),
                  ),
                ],
              ),
            ),
            SizedBox(height: 32),

            // Exercise Type Selection
            Text(
              'Select Exercise Type:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: _selectedExercise,
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              ),
              items: _exerciseTypes.map((exercise) {
                return DropdownMenuItem(
                  value: exercise,
                  child: Text(exercise.replaceAll('_', ' ').toUpperCase()),
                );
              }).toList(),
              onChanged: (value) {
                setState(() {
                  _selectedExercise = value!;
                });
              },
            ),
            SizedBox(height: 24),

            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _recordVideo,
                    icon: Icon(Icons.videocam),
                    label: Text('Record Video'),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickVideo,
                    icon: Icon(Icons.folder),
                    label: Text('Pick Video'),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 16),

            // Video Preview and Analysis Status
            if (_lastVideoPath != null || _isAnalyzing) ...[
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blue[200]!),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.videocam, color: Colors.blue[600]),
                        SizedBox(width: 8),
                        Text(
                          'Video Analysis',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.blue[800],
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 12),
                    
                    if (_isAnalyzing) ...[
                      Row(
                        children: [
                          SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor: AlwaysStoppedAnimation<Color>(Colors.blue[600]!),
                            ),
                          ),
                          SizedBox(width: 12),
                          Text(
                            'Analyzing ${_selectedExercise.replaceAll('_', ' ')} exercise...',
                            style: TextStyle(color: Colors.blue[700]),
                          ),
                        ],
                      ),
                    ] else if (_lastVideoPath != null) ...[
                      Row(
                        children: [
                          Icon(Icons.check_circle, color: Colors.green[600], size: 20),
                          SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'Video recorded successfully',
                              style: TextStyle(color: Colors.green[700]),
                            ),
                          ),
                          TextButton(
                            onPressed: () {
                              setState(() {
                                _lastVideoPath = null;
                                _lastAnalysis = null;
                              });
                            },
                            child: Text('Clear'),
                          ),
                        ],
                      ),
                    ],
                  ],
                ),
              ),
              SizedBox(height: 16),
            ],

            // Get Heatmap Button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _getHeatmap,
                icon: Icon(Icons.thermostat),
                label: Text('Get Torque Heatmap'),
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: 16),
                  backgroundColor: Colors.orange,
                ),
              ),
            ),
            SizedBox(height: 32),

            // Loading Indicator
            if (_isAnalyzing)
              Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text('Analyzing movement...'),
                  ],
                ),
              ),

            // Error Display
            Consumer<BiomechanicsProvider>(
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

            // Analysis Results
            if (_lastAnalysis != null) ...[
              Text(
                'Analysis Results:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 16),
              _buildAnalysisCard(_lastAnalysis!),
            ],

            // Heatmap Display
            if (_lastHeatmap != null) ...[
              SizedBox(height: 24),
              Text(
                'Torque Heatmap:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 16),
              Container(
                width: double.infinity,
                height: 200,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey[300]!),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Center(
                  child: Text('Heatmap visualization would appear here'),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _recordVideo() async {
    try {
      // Navigate to video recorder
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => VideoRecorder(
            exerciseType: _selectedExercise,
            onVideoRecorded: (String videoPath) async {
              // Analyze the recorded video
              await _analyzeVideo(videoPath);
            },
          ),
        ),
      );
    } catch (e) {
      _showError('Error recording video: $e');
    }
  }

  Future<void> _pickVideo() async {
    try {
      final XFile? video = await _picker.pickVideo(source: ImageSource.gallery);
      if (video != null) {
        await _analyzeVideo(video.path);
      }
    } catch (e) {
      _showError('Error picking video: $e');
    }
  }

  Future<void> _analyzeVideo(String videoPath) async {
    setState(() {
      _isAnalyzing = true;
      _lastVideoPath = videoPath;
    });

    try {
      final provider = context.read<BiomechanicsProvider>();
      final analysis = await provider.analyzeMovement(
        videoPath,
        _selectedExercise,
        'default_user',
      );
      
      setState(() {
        _lastAnalysis = analysis;
        _isAnalyzing = false;
      });

      _showAnalysisResults(analysis);
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
      });
      _showError('Error analyzing video: $e');
    }
  }

  Future<void> _getHeatmap() async {
    final provider = context.read<BiomechanicsProvider>();
    final heatmap = await provider.getTorqueHeatmap('default_user', _selectedExercise);
    
    setState(() {
      _lastHeatmap = heatmap;
    });
  }

  void _showAnalysisResults(Map<String, dynamic> analysis) {
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
                  child: _buildAnalysisCard(analysis),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAnalysisCard(Map<String, dynamic> analysis) {
    final analysisData = analysis['analysis'] ?? {};
    final formScore = analysisData['form_score'] ?? 0;
    final issues = List<String>.from(analysisData['issues'] ?? []);
    final recommendations = List<String>.from(analysisData['recommendations'] ?? []);

    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Form Score
            Row(
              children: [
                Icon(Icons.score, color: Colors.blue),
                SizedBox(width: 8),
                Text(
                  'Form Score: $formScore%',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            SizedBox(height: 16),

            // Issues
            if (issues.isNotEmpty) ...[
              Text(
                'Issues Detected:',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 8),
              ...issues.map((issue) => Padding(
                padding: EdgeInsets.only(bottom: 4),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.warning, color: Colors.orange, size: 16),
                    SizedBox(width: 8),
                    Expanded(child: Text(issue)),
                  ],
                ),
              )),
              SizedBox(height: 16),
            ],

            // Recommendations
            if (recommendations.isNotEmpty) ...[
              Text(
                'Recommendations:',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 8),
              ...recommendations.map((rec) => Padding(
                padding: EdgeInsets.only(bottom: 4),
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
          ],
        ),
      ),
    );
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }
}
