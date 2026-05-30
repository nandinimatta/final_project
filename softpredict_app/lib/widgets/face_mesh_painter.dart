import 'package:flutter/material.dart';

import '../models/face_prediction.dart';

class FaceMeshPainter extends CustomPainter {
  final List<FacePoint> points;
  final List<List<int>> connections;
  final Color pointColor;
  final Color lineColor;

  FaceMeshPainter({
    required this.points,
    required this.connections,
    required this.pointColor,
    required this.lineColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (points.isEmpty) {
      return;
    }

    final linePaint = Paint()
      ..color = lineColor
      ..strokeWidth = 1.2
      ..style = PaintingStyle.stroke;

    final pointPaint = Paint()
      ..color = pointColor
      ..style = PaintingStyle.fill;

    final mapped = points
        .map(
          (point) => Offset(
            point.x * size.width,
            point.y * size.height,
          ),
        )
        .toList();

    for (final connection in connections) {
      if (connection.length < 2) {
        continue;
      }

      final startIndex = connection[0];
      final endIndex = connection[1];
      if (startIndex >= mapped.length || endIndex >= mapped.length) {
        continue;
      }

      canvas.drawLine(mapped[startIndex], mapped[endIndex], linePaint);
    }

    for (final offset in mapped) {
      canvas.drawCircle(offset, 1.6, pointPaint);
    }
  }

  @override
  bool shouldRepaint(covariant FaceMeshPainter oldDelegate) {
    return oldDelegate.points != points || oldDelegate.connections != connections;
  }
}
