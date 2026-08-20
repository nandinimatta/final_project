import 'dart:typed_data';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:path_provider/path_provider.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/rendering.dart';

import 'package:flutter/material.dart';

import '../models/face_prediction.dart';
import '../widgets/deformation_painter.dart';
import '../services/api_service.dart';

class PredictionResultScreen extends StatefulWidget {
  final Uint8List imageBytes;
  final FacePrediction prediction;

  const PredictionResultScreen({
    super.key,
    required this.imageBytes,
    required this.prediction,
  });

  @override
  State<PredictionResultScreen> createState() => _PredictionResultScreenState();
}

class _PredictionResultScreenState extends State<PredictionResultScreen> {
  final GlobalKey _afterRepaintKey = GlobalKey();

  Uint8List? _meshImageBytes;
  Uint8List? _afterImageBytes;
  String? _correctionError;
  bool _loadingCorrection = true;

  @override
  void initState() {
    super.initState();
    _loadCorrectedImage();
  }

  Future<void> _loadCorrectedImage() async {
    try {
      // Call the backend /correct endpoint to get all three images
      final images = await ApiService().correctImageThreePanels(
        widget.imageBytes,
        'face_image.jpg',
        method: 'jaw',
      );
      
      if (!mounted) {
        return;
      }

      setState(() {
        _meshImageBytes = images['mesh'];
        _afterImageBytes = images['after'];
        _correctionError = null;
        _loadingCorrection = false;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _correctionError = 'Backend error: $error';
        _loadingCorrection = false;
      });
    }
  }



  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Clinical Correction Result')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _SummaryCard(prediction: widget.prediction),
            const SizedBox(height: 16),
            if (!widget.prediction.modelLoaded)
              const _ModelWarningBanner(),
            if (!widget.prediction.modelLoaded)
              const SizedBox(height: 16),
            const _DiagnosisSummaryCard(),
            const SizedBox(height: 16),
            const _SectionHeader(
              title: 'Clinical Correction Analysis',
              subtitle: 'Before, Mesh Overlay, and After Comparison',
            ),
            const SizedBox(height: 12),
            // Three-image carousel
            SizedBox(
              height: 400,
              child: _loadingCorrection
                  ? const Center(child: CircularProgressIndicator())
                  : _correctionError != null
                      ? Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const Icon(Icons.error_outline, size: 48, color: Colors.red),
                              const SizedBox(height: 8),
                              Text(_correctionError!, textAlign: TextAlign.center),
                            ],
                          ),
                        )
                      : PageView(
                          children: [
                            _ImagePanel(
                              title: 'Before',
                              subtitle: 'Original Image',
                              imageBytes: widget.imageBytes,
                            ),
                            if (_meshImageBytes != null)
                              _ImagePanel(
                                title: 'Mesh Overlay',
                                subtitle: 'Facial Structure Analysis',
                                imageBytes: _meshImageBytes!,
                              ),
                            if (_afterImageBytes != null)
                              _ImagePanel(
                                title: 'After',
                                subtitle: 'Post-correction Preview',
                                imageBytes: _afterImageBytes!,
                              ),
                          ],
                        ),
            ),
            const SizedBox(height: 12),
            if (!_loadingCorrection && _correctionError == null)
              Center(
                child: Text(
                  'Swipe to view: Before → Mesh Overlay → After',
                  style: Theme.of(context).textTheme.bodySmall,
                  textAlign: TextAlign.center,
                ),
              ),
            const SizedBox(height: 16),
            const _GuidanceCard(),
            const SizedBox(height: 12),
            Center(
              child: ElevatedButton.icon(
                onPressed: kIsWeb
                    ? null
                    : () async {
                        final messenger = ScaffoldMessenger.of(context);
                        final saved = await _saveCorrectedImage();
                        messenger.showSnackBar(SnackBar(
                          content: Text(saved ? 'Saved corrected image to temporary folder' : 'Failed to save image'),
                        ));
                      },
                icon: const Icon(Icons.save),
                label: const Text('Save Correction'),
              ),
            ),
            const SizedBox(height: 16),
            const _SectionHeader(
                title: 'Landmark displacement analysis',
                subtitle:
                  'Diagnostic overlay showing the direction and magnitude of soft-tissue landmark movement used by the demo model.',
            ),
            const SizedBox(height: 12),
            _DeformationCard(prediction: widget.prediction),
            const SizedBox(height: 20),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () => Navigator.of(context).popUntil((route) => route.isFirst),
              icon: const Icon(Icons.upload_file),
              label: const Text('Analyze Another Image'),
            ),
          ],
        ),
      ),
    );
  }

  Future<bool> _saveCorrectedImage() async {
    if (_afterImageBytes != null) {
      try {
        if (kIsWeb) return false;
        final dir = await getTemporaryDirectory();
        final file = File('${dir.path}/corrected_${DateTime.now().millisecondsSinceEpoch}.png');
        await file.writeAsBytes(_afterImageBytes!);
        return true;
      } catch (_) {
        return false;
      }
    }

    return _captureAndSave(_afterRepaintKey);
  }
}

class _SummaryCard extends StatelessWidget {
  final FacePrediction prediction;

  const _SummaryCard({required this.prediction});

  @override
  Widget build(BuildContext context) {
    final status = _buildStatus();

    return Card(
      elevation: 2,
      color: const Color(0xFFF8FAFC),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text(
                        'Clinical correction report',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
                      ),
                      SizedBox(height: 4),
                      Text(
                        'AI-assisted maxillofacial landmark analysis',
                        style: TextStyle(fontSize: 13, color: Colors.black54),
                      ),
                    ],
                  ),
                ),
                _StatusBadge(status: status),
              ],
            ),
            const SizedBox(height: 14),
            _ReportHeader(prediction: prediction, statusLabel: status.label),
            const SizedBox(height: 14),
            _ReportLine(label: 'Clinical note', value: 'AI-assisted facial landmark evaluation completed.'),
            _ReportLine(label: 'Landmarks detected', value: '${prediction.landmarkCount}'),
            _ReportLine(label: 'Model confidence', value: '${(prediction.demoScore * 100).toStringAsFixed(1)}%'),
            _ReportLine(label: 'Model status', value: prediction.modelLoaded ? 'Loaded' : 'Demo fallback active'),
            _ReportLine(label: 'Mean displacement', value: prediction.averageShift.toStringAsFixed(4)),
            _ReportLine(label: 'Peak displacement', value: prediction.maxShift.toStringAsFixed(4)),
            _ReportLine(label: 'Affected points', value: '${prediction.movingPoints}'),
          ],
        ),
      ),
    );
  }

  _BadgeStatus _buildStatus() {
    if (prediction.averageShift < 0.02 && prediction.demoScore >= 0.9) {
      return const _BadgeStatus(label: 'LOW RISK', background: Color(0xFFD1FAE5), foreground: Color(0xFF065F46));
    }

    if (prediction.averageShift < 0.05) {
      return const _BadgeStatus(label: 'MODERATE', background: Color(0xFFFEF3C7), foreground: Color(0xFF92400E));
    }

    return const _BadgeStatus(label: 'IMPROVED', background: Color(0xFFDBEAFE), foreground: Color(0xFF1D4ED8));
  }
}

class _StatusBadge extends StatelessWidget {
  final _BadgeStatus status;

  const _StatusBadge({required this.status});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: status.background,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: status.foreground.withValues(alpha: 0.18)),
      ),
      child: Text(
        status.label,
        style: TextStyle(
          color: status.foreground,
          fontSize: 12,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.4,
        ),
      ),
    );
  }
}

class _BadgeStatus {
  final String label;
  final Color background;
  final Color foreground;

  const _BadgeStatus({
    required this.label,
    required this.background,
    required this.foreground,
  });
}

class _ReportHeader extends StatelessWidget {
  final FacePrediction prediction;
  final String statusLabel;

  const _ReportHeader({required this.prediction, required this.statusLabel});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Clinical metadata',
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 14,
            runSpacing: 10,
            children: [
              _MetaChip(label: 'Study ID', value: 'SOFT-${prediction.landmarkCount}'),
              _MetaChip(label: 'Status', value: statusLabel),
              _MetaChip(label: 'Modality', value: 'Facial photograph'),
              _MetaChip(label: 'Engine', value: prediction.modelLoaded ? 'Loaded' : 'Demo fallback'),
            ],
          ),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  final String subtitle;

  const _SectionHeader({required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 6),
        Text(
          subtitle,
          style: const TextStyle(fontSize: 14, color: Colors.black87, height: 1.4),
        ),
      ],
    );
  }
}



Future<bool> _captureAndSave(GlobalKey key) async {
  try {
    if (kIsWeb) return false;
    final boundary = key.currentContext?.findRenderObject() as RenderRepaintBoundary?;
    if (boundary == null) return false;
    final devicePixelRatio = ui.PlatformDispatcher.instance.views.first.devicePixelRatio;
    final image = await boundary.toImage(pixelRatio: devicePixelRatio);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) return false;
    final bytes = byteData.buffer.asUint8List();

    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/corrected_${DateTime.now().millisecondsSinceEpoch}.png');
    await file.writeAsBytes(bytes);
    return true;
  } catch (e) {
    return false;
  }
}

class _MetaChip extends StatelessWidget {
  final String label;
  final String value;

  const _MetaChip({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFF8FAFC),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: const TextStyle(fontSize: 11, color: Colors.black54, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

class _ReportLine extends StatelessWidget {
  final String label;
  final String value;

  const _ReportLine({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 142,
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w700,
                color: Color(0xFF334155),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontSize: 13, height: 1.35),
            ),
          ),
        ],
      ),
    );
  }
}

class JawRefinementPainter extends CustomPainter {
  final List<FacePoint> points;

  JawRefinementPainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 17) {
      return;
    }

    final jawLeft = _map(points[3], size);
    final jawRight = _map(points[13], size);
    final chin = _map(points[8], size);

    final jawWidth = (jawRight.dx - jawLeft.dx).abs();
    final jawRect = Rect.fromCenter(
      center: Offset(chin.dx, chin.dy + size.height * 0.03),
      width: jawWidth * 1.18,
      height: size.height * 0.33,
    );

    final softMask = Paint()
      ..shader = ui.Gradient.radial(
        jawRect.center,
        jawRect.width * 0.58,
        [
          const Color(0xFFF8E3D8).withValues(alpha: 0.42),
          const Color(0xFFF8E3D8).withValues(alpha: 0.10),
          Colors.transparent,
        ],
        [0.0, 0.62, 1.0],
      )
      ..blendMode = BlendMode.softLight;

    canvas.drawOval(jawRect, softMask);

    final highlightPath = Path()
      ..moveTo(jawLeft.dx, jawLeft.dy)
      ..quadraticBezierTo(chin.dx - jawWidth * 0.2, chin.dy + size.height * 0.01, chin.dx, chin.dy + size.height * 0.03)
      ..quadraticBezierTo(chin.dx + jawWidth * 0.2, chin.dy + size.height * 0.01, jawRight.dx, jawRight.dy);

    final correctionPaint = Paint()
      ..color = const Color(0xFF34D399).withValues(alpha: 0.95)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.6
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    canvas.drawPath(highlightPath, correctionPaint);

    final innerGlow = Paint()
      ..color = const Color(0xFFA7F3D0).withValues(alpha: 0.14)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    canvas.drawPath(highlightPath, innerGlow);
  }

  Offset _map(FacePoint point, Size size) => Offset(point.x * size.width, point.y * size.height);

  @override
  bool shouldRepaint(covariant JawRefinementPainter oldDelegate) {
    return oldDelegate.points != points;
  }
}

class _DeformationCard extends StatelessWidget {
  final FacePrediction prediction;

  const _DeformationCard({required this.prediction});

  @override
  Widget build(BuildContext context) {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            color: const Color(0xFF1E293B),
            child: const Text(
              'Landmark Motion Overlay',
              style: TextStyle(color: Colors.white, fontWeight: FontWeight.w700),
            ),
          ),
          AspectRatio(
            aspectRatio: 1,
            child: CustomPaint(
              painter: DeformationPainter(
                beforePoints: prediction.landmarks,
                afterPoints: prediction.predictedLandmarks,
                connections: prediction.meshConnections,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: Wrap(
              spacing: 12,
              runSpacing: 8,
              children: const [
                _LegendDot(color: Color(0xFF14B8A6), label: 'Before'),
                _LegendDot(color: Color(0xFFF97316), label: 'After'),
                _LegendDot(color: Color(0xFFEF4444), label: 'Movement'),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;

  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(label),
      ],
    );
  }
}

class _GuidanceCard extends StatelessWidget {
  const _GuidanceCard();

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1.5,
      color: const Color(0xFFF8FAFC),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Text(
              'Clinical capture and recovery guidance',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
            ),
            SizedBox(height: 10),
            _ChecklistRow(
              icon: Icons.check_circle_outline,
              iconColor: Color(0xFF059669),
              text: 'Capture a frontal, neutral-expression photograph with the lower face fully visible.',
            ),
            _ChecklistRow(
              icon: Icons.check_circle_outline,
              iconColor: Color(0xFF059669),
              text: 'Use even illumination and minimise shadowing over the perioral region.',
            ),
            _ChecklistRow(
              icon: Icons.check_circle_outline,
              iconColor: Color(0xFF059669),
              text: 'Remove filters, masks, lipstick, and cosmetic occlusion before upload.',
            ),
            SizedBox(height: 10),
            _ChecklistRow(
              icon: Icons.cancel_outlined,
              iconColor: Color(0xFFDC2626),
              text: 'Do not submit blurred, tilted, or heavily cropped images.',
            ),
            _ChecklistRow(
              icon: Icons.cancel_outlined,
              iconColor: Color(0xFFDC2626),
              text: 'Do not use the preview as a substitute for a clinician-led surgical plan.',
            ),
            _ChecklistRow(
              icon: Icons.cancel_outlined,
              iconColor: Color(0xFFDC2626),
              text: 'Do not alter medication, diet, or recovery instructions without medical advice.',
            ),
          ],
        ),
      ),
    );
  }
}

class _ChecklistRow extends StatelessWidget {
  final IconData icon;
  final Color iconColor;
  final String text;

  const _ChecklistRow({required this.icon, required this.iconColor, required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 22,
            height: 22,
            alignment: Alignment.center,
            child: Icon(
              icon,
              size: 20,
              color: iconColor,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(height: 1.35),
            ),
          ),
        ],
      ),
    );
  }
}

class _DiagnosisSummaryCard extends StatelessWidget {
  const _DiagnosisSummaryCard();

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 1.5,
      color: const Color(0xFFF0FDF4),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Row(
              children: [
                Icon(Icons.medical_information_outlined, color: Color(0xFF166534)),
                SizedBox(width: 8),
                Text(
                  'Orthognathic correction preview',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
                ),
              ],
            ),
            SizedBox(height: 10),
            Text('Assessment: oral and lower-facial landmarks detected with visible contour variation.'),
            Text('Procedure class: simulated orthognathic / maxillofacial correction preview.'),
            Text('Outcome: the post-correction image is rendered as the refined version of the same face.'),
          ],
        ),
      ),
    );
  }
}

class _ModelWarningBanner extends StatelessWidget {
  const _ModelWarningBanner();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF7ED),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFF59E0B).withValues(alpha: 0.35)),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.warning_amber_rounded, color: Color(0xFFD97706)),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'Model weights file not loaded. The app is using fallback correction behavior until softpredict_model.pth is available on the backend.',
              style: TextStyle(fontSize: 13.5, height: 1.35, color: Color(0xFF92400E)),
            ),
          ),
        ],
      ),
    );
  }
}

class _ImagePanel extends StatelessWidget {
  final String title;
  final String subtitle;
  final Uint8List imageBytes;

  const _ImagePanel({
    required this.title,
    required this.subtitle,
    required this.imageBytes,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFE2E8F0)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: const BoxDecoration(
              color: Color(0xFFF8FAFC),
              borderRadius: BorderRadius.vertical(top: Radius.circular(14)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                    color: Color(0xFF1E293B),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(
                    fontSize: 13,
                    color: Color(0xFF64748B),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: const BorderRadius.vertical(bottom: Radius.circular(14)),
              child: Image.memory(
                imageBytes,
                fit: BoxFit.contain,
                alignment: Alignment.center,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
