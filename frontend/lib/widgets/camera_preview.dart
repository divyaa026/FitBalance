import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class CameraPreview extends StatelessWidget {
  final CameraController controller;
  final VoidCallback? onTap;

  const CameraPreview({
    Key? key,
    required this.controller,
    this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        height: double.infinity,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: CameraPreview(controller),
        ),
      ),
    );
  }
} 