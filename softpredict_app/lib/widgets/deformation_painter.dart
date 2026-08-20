import 'package:flutter/material.dart';
import '../models/face_prediction.dart';

class DeformationPainter extends CustomPainter {
  final List<FacePoint> beforePoints;
  final List<FacePoint> afterPoints;
  final List<List<int>> connections;

  DeformationPainter({
    required this.beforePoints,
    required this.afterPoints,
    required this.connections,
  });

  // Jawline and outer lips landmarks to display a clean clinical profile
  static final List<int> jawIndices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
  
  static final List<int> mouthLoop = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 291, 
    321, 314, 317, 318, 402, 312, 311, 310, 415, 
    95, 88, 178, 87, 14, 13, 82, 81, 80, 191, 61
  ];

  static final Set<int> lowerFaceIndices = {...jawIndices, ...mouthLoop};

  @override
  void paint(Canvas canvas, Size size) {
    if (beforePoints.isEmpty || afterPoints.isEmpty) {
      return;
    }

    // 1. Calculate bounding box of the active lower face landmarks
    double minX = double.infinity;
    double maxX = -double.infinity;
    double minY = double.infinity;
    double maxY = -double.infinity;

    for (final idx in lowerFaceIndices) {
      if (idx < beforePoints.length) {
        final ptB = beforePoints[idx];
        if (ptB.x < minX) minX = ptB.x;
        if (ptB.x > maxX) maxX = ptB.x;
        if (ptB.y < minY) minY = ptB.y;
        if (ptB.y > maxY) maxY = ptB.y;
      }
      if (idx < afterPoints.length) {
        final ptA = afterPoints[idx];
        if (ptA.x < minX) minX = ptA.x;
        if (ptA.x > maxX) maxX = ptA.x;
        if (ptA.y < minY) minY = ptA.y;
        if (ptA.y > maxY) maxY = ptA.y;
      }
    }

    if (minX == double.infinity) {
      minX = 0.0; maxX = 1.0;
      minY = 0.0; maxY = 1.0;
    }

    final boxWidth = maxX - minX;
    final boxHeight = maxY - minY;

    final centerX = minX + boxWidth / 2;
    final centerY = minY + boxHeight / 2;

    // Apply a scaling factor with 20% margin to prevent screen boundary clipping
    final scaleX = (size.width * 0.8) / boxWidth;
    final scaleY = (size.height * 0.8) / boxHeight;
    final scale = scaleX < scaleY ? scaleX : scaleY;

    Offset mapPoint(FacePoint point) {
      final x = size.width / 2 + (point.x - centerX) * scale;
      final y = size.height / 2 + (point.y - centerY) * scale;
      return Offset(x, y);
    }

    // 2. Define paints
    // Thick, semi-transparent red band for displacement movement
    final movementPaint = Paint()
      ..color = const Color(0x52EF4444)
      ..strokeWidth = 22.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final linePaintBefore = Paint()
      ..color = const Color(0x6614B8A6)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final linePaintAfter = Paint()
      ..color = const Color(0x66F97316)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final dotPaintBefore = Paint()
      ..color = const Color(0xFF14B8A6)
      ..style = PaintingStyle.fill;

    final dotPaintAfter = Paint()
      ..color = const Color(0xFFF97316)
      ..style = PaintingStyle.fill;

    // 3. Draw displacement movement red bands (connect before and after for each node)
    for (final idx in lowerFaceIndices) {
      if (idx < beforePoints.length && idx < afterPoints.length) {
        final beforeOffset = mapPoint(beforePoints[idx]);
        final afterOffset = mapPoint(afterPoints[idx]);
        canvas.drawLine(beforeOffset, afterOffset, movementPaint);
      }
    }

    // 4. Draw Before & After outline curves
    final beforeJawPoints = jawIndices.where((idx) => idx < beforePoints.length).map((idx) => mapPoint(beforePoints[idx])).toList();
    final afterJawPoints = jawIndices.where((idx) => idx < afterPoints.length).map((idx) => mapPoint(afterPoints[idx])).toList();

    for (int i = 0; i < beforeJawPoints.length - 1; i++) {
      canvas.drawLine(beforeJawPoints[i], beforeJawPoints[i + 1], linePaintBefore);
      canvas.drawLine(afterJawPoints[i], afterJawPoints[i + 1], linePaintAfter);
    }

    final beforeMouthPoints = mouthLoop.where((idx) => idx < beforePoints.length).map((idx) => mapPoint(beforePoints[idx])).toList();
    final afterMouthPoints = mouthLoop.where((idx) => idx < afterPoints.length).map((idx) => mapPoint(afterPoints[idx])).toList();

    for (int i = 0; i < beforeMouthPoints.length - 1; i++) {
      canvas.drawLine(beforeMouthPoints[i], beforeMouthPoints[i + 1], linePaintBefore);
      canvas.drawLine(afterMouthPoints[i], afterMouthPoints[i + 1], linePaintAfter);
    }

    // 5. Draw node circles (cyan for before, orange for after)
    for (final idx in lowerFaceIndices) {
      if (idx < beforePoints.length) {
        canvas.drawCircle(mapPoint(beforePoints[idx]), 3.5, dotPaintBefore);
      }
      if (idx < afterPoints.length) {
        canvas.drawCircle(mapPoint(afterPoints[idx]), 3.5, dotPaintAfter);
      }
    }
  }

  @override
  bool shouldRepaint(covariant DeformationPainter oldDelegate) {
    return oldDelegate.beforePoints != beforePoints ||
        oldDelegate.afterPoints != afterPoints ||
        oldDelegate.connections != connections;
  }
}
