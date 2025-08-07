import 'package:flutter/material.dart';
import 'dart:math' as math;

class ProteinGauge extends StatelessWidget {
  final double protein;
  final double targetProtein;
  final double size;

  const ProteinGauge({
    Key? key,
    required this.protein,
    required this.targetProtein,
    this.size = 200,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final progress = (protein / targetProtein).clamp(0.0, 1.0);
    final percentage = (progress * 100).round();
    
    return Container(
      width: size,
      height: size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Background circle
          CustomPaint(
            size: Size(size, size),
            painter: ProteinGaugePainter(
              progress: 0,
              color: Colors.grey[300]!,
              strokeWidth: 12,
            ),
          ),
          
          // Progress circle
          CustomPaint(
            size: Size(size, size),
            painter: ProteinGaugePainter(
              progress: progress,
              color: _getProgressColor(progress),
              strokeWidth: 12,
            ),
          ),
          
          // Center content
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                '${protein.round()}g',
                style: TextStyle(
                  fontSize: size * 0.15,
                  fontWeight: FontWeight.bold,
                  color: _getProgressColor(progress),
                ),
              ),
              Text(
                'of ${targetProtein.round()}g',
                style: TextStyle(
                  fontSize: size * 0.08,
                  color: Colors.grey[600],
                ),
              ),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: _getProgressColor(progress).withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '$percentage%',
                  style: TextStyle(
                    fontSize: size * 0.06,
                    fontWeight: FontWeight.bold,
                    color: _getProgressColor(progress),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Color _getProgressColor(double progress) {
    if (progress < 0.5) {
      return Colors.red;
    } else if (progress < 0.8) {
      return Colors.orange;
    } else if (progress < 1.0) {
      return Colors.green;
    } else {
      return Colors.blue;
    }
  }
}

class ProteinGaugePainter extends CustomPainter {
  final double progress;
  final Color color;
  final double strokeWidth;

  ProteinGaugePainter({
    required this.progress,
    required this.color,
    required this.strokeWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;
    
    // Background circle
    final backgroundPaint = Paint()
      ..color = Colors.grey[300]!
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;
    
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2, // Start from top
      2 * math.pi, // Full circle
      false,
      backgroundPaint,
    );
    
    // Progress circle
    final progressPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;
    
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2, // Start from top
      2 * math.pi * progress, // Progress portion
      false,
      progressPaint,
    );
  }

  @override
  bool shouldRepaint(ProteinGaugePainter oldDelegate) {
    return oldDelegate.progress != progress ||
           oldDelegate.color != color ||
           oldDelegate.strokeWidth != strokeWidth;
  }
} 