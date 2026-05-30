class FacePoint {
  final double x;
  final double y;
  final double z;

  const FacePoint({required this.x, required this.y, required this.z});

  factory FacePoint.fromJson(Map<String, dynamic> json) {
    return FacePoint(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      z: (json['z'] as num).toDouble(),
    );
  }
}

class FacePrediction {
  final bool modelLoaded;
  final double demoScore;
  final int landmarkCount;
  final double averageShift;
  final double maxShift;
  final int movingPoints;
  final String message;
  final List<FacePoint> landmarks;
  final List<FacePoint> predictedLandmarks;
  final List<List<int>> meshConnections;

  const FacePrediction({
    required this.modelLoaded,
    required this.demoScore,
    required this.landmarkCount,
    required this.averageShift,
    required this.maxShift,
    required this.movingPoints,
    required this.message,
    required this.landmarks,
    required this.predictedLandmarks,
    required this.meshConnections,
  });

  factory FacePrediction.fromJson(Map<String, dynamic> json) {
    return FacePrediction(
      modelLoaded: json['model_loaded'] as bool? ?? false,
      demoScore: (json['demo_score'] as num?)?.toDouble() ?? 0.0,
      landmarkCount: (json['landmark_count'] as num?)?.toInt() ?? 0,
        averageShift: (json['deformation_summary']?['average_shift'] as num?)?.toDouble() ?? 0.0,
        maxShift: (json['deformation_summary']?['max_shift'] as num?)?.toDouble() ?? 0.0,
        movingPoints: (json['deformation_summary']?['moving_points'] as num?)?.toInt() ?? 0,
      message: json['message'] as String? ?? '',
      landmarks: (json['landmarks'] as List<dynamic>? ?? [])
          .map((point) => FacePoint.fromJson(point as Map<String, dynamic>))
          .toList(),
      predictedLandmarks: (json['predicted_landmarks'] as List<dynamic>? ?? [])
          .map((point) => FacePoint.fromJson(point as Map<String, dynamic>))
          .toList(),
      meshConnections: (json['mesh_connections'] as List<dynamic>? ?? [])
          .map(
            (pair) => (pair as List<dynamic>)
                .map((value) => (value as num).toInt())
                .toList(),
          )
          .toList(),
    );
  }
}
