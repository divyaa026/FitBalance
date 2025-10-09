import 'package:flutter/material.dart';

class ShapExplanationBubble extends StatefulWidget {
  final String message;
  final Duration animationDuration;
  final bool showIcon;

  const ShapExplanationBubble({
    Key? key,
    required this.message,
    this.animationDuration = const Duration(milliseconds: 300),
    this.showIcon = true,
  }) : super(key: key);

  @override
  _ShapExplanationBubbleState createState() => _ShapExplanationBubbleState();
}

class _ShapExplanationBubbleState extends State<ShapExplanationBubble>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _scaleAnimation;
  late Animation<double> _opacityAnimation;
  bool _isExpanded = false;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: widget.animationDuration,
      vsync: this,
    );
    
    _scaleAnimation = Tween<double>(
      begin: 0.8,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOutBack,
    ));
    
    _opacityAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeIn,
    ));
    
    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  void _toggleExpanded() {
    setState(() {
      _isExpanded = !_isExpanded;
    });
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animationController,
      builder: (context, child) {
        return Transform.scale(
          scale: _scaleAnimation.value,
          child: Opacity(
            opacity: _opacityAnimation.value,
            child: GestureDetector(
              onTap: _toggleExpanded,
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      Colors.purple[50]!,
                      Colors.purple[100]!,
                    ],
                  ),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: Colors.purple[200]!,
                    width: 1.5,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.purple[200]!.withOpacity(0.3),
                      blurRadius: 8,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Header
                    Row(
                      children: [
                        if (widget.showIcon) ...[
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Colors.purple[600],
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: const Icon(
                              Icons.psychology,
                              color: Colors.white,
                              size: 16,
                            ),
                          ),
                          const SizedBox(width: 12),
                        ],
                        Expanded(
                          child: Text(
                            'AI Explanation',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.purple[800],
                              fontSize: 16,
                            ),
                          ),
                        ),
                        Icon(
                          _isExpanded ? Icons.expand_less : Icons.expand_more,
                          color: Colors.purple[600],
                        ),
                      ],
                    ),
                    
                    const SizedBox(height: 8),
                    
                    // Message
                    AnimatedCrossFade(
                      duration: const Duration(milliseconds: 200),
                      crossFadeState: _isExpanded 
                          ? CrossFadeState.showSecond 
                          : CrossFadeState.showFirst,
                      firstChild: Text(
                        widget.message.length > 100 
                            ? '${widget.message.substring(0, 100)}...'
                            : widget.message,
                        style: TextStyle(
                          color: Colors.purple[700],
                          fontSize: 14,
                          height: 1.4,
                        ),
                      ),
                      secondChild: Text(
                        widget.message,
                        style: TextStyle(
                          color: Colors.purple[700],
                          fontSize: 14,
                          height: 1.4,
                        ),
                      ),
                    ),
                    
                    // SHAP Feature Importance (if expanded)
                    if (_isExpanded) ...[
                      const SizedBox(height: 12),
                      _buildShapFeatures(),
                    ],
                    
                    // Action buttons
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        TextButton(
                          onPressed: () {
                            // TODO: Show detailed SHAP analysis
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Detailed SHAP analysis coming soon!'),
                              ),
                            );
                          },
                          child: Text(
                            'Learn More',
                            style: TextStyle(
                              color: Colors.purple[600],
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        ElevatedButton(
                          onPressed: () {
                            // TODO: Apply recommendation
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Recommendation applied!'),
                                backgroundColor: Colors.green,
                              ),
                            );
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.purple[600],
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                          ),
                          child: const Text('Apply'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildShapFeatures() {
    // Mock SHAP feature importance data
    final features = [
      {'name': 'Sleep Quality', 'importance': 0.35, 'color': Colors.blue},
      {'name': 'Previous Meals', 'importance': 0.28, 'color': Colors.green},
      {'name': 'Activity Level', 'importance': 0.22, 'color': Colors.orange},
      {'name': 'Stress Level', 'importance': 0.15, 'color': Colors.red},
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Feature Importance',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Colors.purple[800],
            fontSize: 14,
          ),
        ),
        const SizedBox(height: 8),
        ...features.map((feature) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 2),
          child: Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  color: feature['color'] as Color,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  feature['name'] as String,
                  style: TextStyle(
                    color: Colors.purple[700],
                    fontSize: 12,
                  ),
                ),
              ),
              Text(
                '${((feature['importance'] as double) * 100).toStringAsFixed(0)}%',
                style: TextStyle(
                  color: Colors.purple[600],
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        )).toList(),
      ],
    );
  }
} 