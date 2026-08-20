import 'dart:convert';
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/face_prediction.dart';
import '../widgets/deformation_painter.dart';

class PatientDetailsScreen extends StatefulWidget {
  final dynamic record;
  const PatientDetailsScreen({super.key, required this.record});

  @override
  State<PatientDetailsScreen> createState() => _PatientDetailsScreenState();
}

class _PatientDetailsScreenState extends State<PatientDetailsScreen> {
  int _activePanelIndex = 0;
  final PageController _pageController = PageController();

  Future<void> _confirmDelete(BuildContext context, int id) async {
    final bool? confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) {
        return AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: const Text('Delete Patient Record?',
              style: TextStyle(fontWeight: FontWeight.bold, color: Color(0xFF0F172A))),
          content: const Text(
            'Are you sure you want to permanently delete this clinical record and all associated visualization images? This action cannot be undone.',
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Cancel',
                  style: TextStyle(color: Color(0xFF64748B))),
              onPressed: () => Navigator.of(ctx).pop(false),
            ),
            TextButton(
              child: const Text('Delete',
                  style: TextStyle(
                      color: Color(0xFFDC2626), fontWeight: FontWeight.bold)),
              onPressed: () => Navigator.of(ctx).pop(true),
            ),
          ],
        );
      },
    );

    if (confirm != true || !mounted) return;
    _deleteRecordWithLoader(id);
  }

  Future<void> _deleteRecordWithLoader(int id) async {
    if (!mounted) return;
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);

    // Show progress loader
    showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => const Center(
          child: CircularProgressIndicator(color: Color(0xFF0F766E))),
    );

    try {
      await ApiService().deleteRecord(id);
      if (!mounted) return;

      navigator.pop(); // Dismiss progress loader

      messenger.showSnackBar(
        const SnackBar(
          content: Text('Clinical record successfully deleted.'),
          backgroundColor: Color(0xFF059669),
        ),
      );

      navigator.pop(true); // Pop details screen and return true to refresh dashboard
    } catch (e) {
      if (!mounted) return;
      navigator.pop(); // Dismiss progress loader
      messenger.showSnackBar(
        SnackBar(
          content: Text('Failed to delete record: $e'),
          backgroundColor: const Color(0xFFDC2626),
        ),
      );
    }
  }

  String _resolveImageUrl(String originalUrl) {
    if (originalUrl.isEmpty) return '';
    try {
      final originalUri = Uri.parse(originalUrl);
      final baseUri = Uri.parse(ApiService.baseUrl);
      final resolvedUri = originalUri.replace(
        scheme: baseUri.scheme,
        host: baseUri.host,
        port: baseUri.port,
      );
      return resolvedUri.toString();
    } catch (e) {
      return originalUrl;
    }
  }

  @override
  Widget build(BuildContext context) {
    final record = widget.record;
    final patientId = record['patient_id'] ?? 'N/A';
    final name = record['name'] ?? 'Unknown';
    final problem = record['problem'] ?? 'No description';
    final treatmentMethod =
        (record['treatment_method'] ?? 'teeth').toString().toLowerCase();
    final age = record['age'] ?? 'N/A';
    final dob = record['dob'] ?? 'N/A';
    final gender = record['gender'] ?? 'N/A';
    final createdAt = record['created_at'] ?? 'N/A';

    final beforeUrl = _resolveImageUrl(record['before_url'] ?? '');
    final meshUrl = _resolveImageUrl(record['mesh_url'] ?? '');
    final afterUrl = _resolveImageUrl(record['after_url'] ?? '');
    final graphUrl = _resolveImageUrl(record['graph_url'] ?? '');

    final indicatedProcedure =
        record['indicated_procedure'] ?? 'Maxillofacial refinement indicated';
    final pathologySummary =
        record['pathology_summary'] ?? 'Skeletal structural variation';

    // Parse Guidelines (Dos and Don'ts)
    List<dynamic> dos = [];
    List<dynamic> donts = [];
    if (record['guidelines_json'] != null &&
        record['guidelines_json'].toString().isNotEmpty) {
      try {
        final Map<String, dynamic> guidelines =
            jsonDecode(record['guidelines_json'] as String)
                as Map<String, dynamic>;
        dos = guidelines['dos'] as List<dynamic>? ?? [];
        donts = guidelines['donts'] as List<dynamic>? ?? [];
      } catch (e) {
        debugPrint('Error parsing guidelines: $e');
      }
    }

    // Parse Landmark coordinate sets
    List<FacePoint> beforePoints = [];
    List<FacePoint> afterPoints = [];
    if (record['landmarks_json'] != null &&
        record['landmarks_json'].toString().isNotEmpty) {
      try {
        final decoded =
            jsonDecode(record['landmarks_json'] as String) as List<dynamic>;
        beforePoints = decoded
            .map((p) => FacePoint.fromJson(p as Map<String, dynamic>))
            .toList();
      } catch (e) {
        debugPrint('Error parsing before coordinates: $e');
      }
    }
    if (record['predicted_landmarks_json'] != null &&
        record['predicted_landmarks_json'].toString().isNotEmpty) {
      try {
        final decoded = jsonDecode(record['predicted_landmarks_json'] as String)
            as List<dynamic>;
        afterPoints = decoded
            .map((p) => FacePoint.fromJson(p as Map<String, dynamic>))
            .toList();
      } catch (e) {
        debugPrint('Error parsing after coordinates: $e');
      }
    }

    // List of panels to display in the carousel
    final List<Map<String, String>> panels = [
      if (beforeUrl.isNotEmpty)
        {
          'title': 'Before (Original)',
          'url': beforeUrl,
          'desc': 'Original pre-treatment profile'
        },
      if (meshUrl.isNotEmpty)
        {
          'title': 'Mesh Overlay',
          'url': meshUrl,
          'desc': 'GCN-predicted structural landmark mesh'
        },
      if (afterUrl.isNotEmpty)
        {
          'title': 'After (Refined)',
          'url': afterUrl,
          'desc': 'Orthopedic post-treatment simulation'
        },
      if (graphUrl.isNotEmpty)
        {
          'title': '3D Shape Simulation',
          'url': graphUrl,
          'desc': '3D depth mapping profile'
        },
    ];

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0F766E),
        foregroundColor: Colors.white,
        title: const Text(
          'Patient Record Details',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_forever_rounded,
                color: Color(0xFFFECACA)),
            tooltip: 'Delete Record',
            onPressed: () => _confirmDelete(context, record['id'] as int),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Patient Core Profile Card
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(18)),
              color: Colors.white,
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Text(
                            name,
                            style: const TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF0F172A),
                            ),
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: treatmentMethod == 'teeth'
                                ? const Color(0xFFECFDF5)
                                : const Color(0xFFEFF6FF),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            treatmentMethod == 'teeth'
                                ? 'ORTHODONTIC'
                                : 'ORTHOGNATHIC',
                            style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w800,
                              color: treatmentMethod == 'teeth'
                                  ? const Color(0xFF059669)
                                  : const Color(0xFF2563EB),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    const Divider(),
                    const SizedBox(height: 12),
                    _buildProfileRow(
                        Icons.badge_outlined, 'Hospital ID / MRN', patientId),
                    _buildProfileRow(Icons.cake_outlined, 'Date of Birth (Age)',
                        '$dob ($age years)'),
                    _buildProfileRow(
                        Icons.wc_outlined, 'Sex & Gender Identity', gender),
                    _buildProfileRow(Icons.calendar_month_outlined,
                        'Record Created', createdAt),
                    const SizedBox(height: 16),
                    const Text(
                      'Primary Diagnosis / Problem',
                      style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF64748B)),
                    ),
                    const SizedBox(height: 6),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: const Color(0xFFF1F5F9),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: const Color(0xFFE2E8F0)),
                      ),
                      child: Text(
                        problem,
                        style: const TextStyle(
                          fontSize: 14.5,
                          color: Color(0xFF334155),
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),

            // Clinical Visualizations Section
            const Text(
              'Clinical Image Visualizations',
              style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF0F172A)),
            ),
            const SizedBox(height: 12),

            // Carousel Slider
            if (panels.isNotEmpty) ...[
              Card(
                elevation: 3,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20)),
                clipBehavior: Clip.antiAlias,
                color: Colors.white,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 12),
                      color: const Color(0xFFF8FAFC),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            panels[_activePanelIndex]['title']!,
                            style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1E293B),
                            ),
                          ),
                          Text(
                            '${_activePanelIndex + 1}/${panels.length}',
                            style: const TextStyle(
                              fontSize: 13,
                              color: Color(0xFF64748B),
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      height: 420,
                      decoration: const BoxDecoration(
                        color: Color(0xFF0F172A),
                      ),
                      alignment: Alignment.center,
                      child: PageView.builder(
                        controller: _pageController,
                        onPageChanged: (index) {
                          setState(() {
                            _activePanelIndex = index;
                          });
                        },
                        itemCount: panels.length,
                        itemBuilder: (context, index) {
                          final panel = panels[index];
                          return Center(
                            child: Image.network(
                              panel['url']!,
                              fit: BoxFit.contain,
                              alignment: Alignment.center,
                              loadingBuilder:
                                  (context, child, loadingProgress) {
                                if (loadingProgress == null) return child;
                                return const Center(
                                  child: CircularProgressIndicator(
                                      color: Color(0xFF0F766E)),
                                );
                              },
                              errorBuilder: (context, error, stackTrace) {
                                return Center(
                                  child: Padding(
                                    padding: const EdgeInsets.all(24.0),
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Icon(Icons.broken_image_outlined,
                                            size: 48, color: Colors.grey[400]),
                                        const SizedBox(height: 12),
                                        Text(
                                          'Error loading image: ${panel['title']}',
                                          textAlign: TextAlign.center,
                                          style: const TextStyle(
                                              color: Color(0xFF94A3B8),
                                              fontSize: 13),
                                        ),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
                          );
                        },
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.all(12),
                      color: const Color(0xFFF1F5F9),
                      child: Text(
                        panels[_activePanelIndex]['desc']!,
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                            fontSize: 13, color: Color(0xFF475569)),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(panels.length, (index) {
                  return InkWell(
                    onTap: () {
                      _pageController.animateToPage(
                        index,
                        duration: const Duration(milliseconds: 300),
                        curve: Curves.easeInOut,
                      );
                    },
                    child: Container(
                      width: 8,
                      height: 8,
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: _activePanelIndex == index
                            ? const Color(0xFF0F766E)
                            : const Color(0xFFCBD5E1),
                      ),
                    ),
                  );
                }),
              ),
            ],
            const SizedBox(height: 24),

            // Landmark Motion Overlay Card (From second image)
            if (beforePoints.isNotEmpty && afterPoints.isNotEmpty) ...[
              const Text(
                'Landmark Displacement Analysis',
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF0F172A)),
              ),
              const SizedBox(height: 6),
              const Text(
                'Diagnostic overlay showing the direction and magnitude of soft-tissue landmark movement.',
                style: TextStyle(
                    fontSize: 13.5, color: Color(0xFF64748B), height: 1.35),
              ),
              const SizedBox(height: 12),
              Card(
                elevation: 2,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20)),
                clipBehavior: Clip.antiAlias,
                color: Colors.white,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(14),
                      color: const Color(0xFF0F172A),
                      child: const Text(
                        'Landmark Motion Overlay',
                        style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 14),
                      ),
                    ),
                    Container(
                      height: 320,
                      color: const Color(0xFFF8FAFC),
                      child: CustomPaint(
                        painter: DeformationPainter(
                          beforePoints: beforePoints,
                          afterPoints: afterPoints,
                          connections: const [],
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          vertical: 12, horizontal: 16),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          _buildLegendDot(const Color(0xFF14B8A6), 'Before'),
                          const SizedBox(width: 24),
                          _buildLegendDot(const Color(0xFFF97316), 'After'),
                          const SizedBox(width: 24),
                          _buildLegendDot(
                              const Color(0x99EF4444),
                              'Movement'),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
            ],

            // Clinical Diagnosis & Surgical Treatment Plan Card
            const Text(
              'Surgical Treatment & Guidelines Plan',
              style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF0F172A)),
            ),
            const SizedBox(height: 12),
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(18)),
              color: Colors.white,
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Indicated surgical procedure
                    const Row(
                      children: [
                        Icon(Icons.assignment_outlined,
                            color: Color(0xFF0F766E), size: 22),
                        SizedBox(width: 8),
                        Text(
                          'Indicated Surgical Procedure',
                          style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF0F172A)),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      indicatedProcedure,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF0F766E),
                        height: 1.3,
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Divider(),
                    const SizedBox(height: 16),

                    // Pathology summary
                    const Row(
                      children: [
                        Icon(Icons.medical_information_outlined,
                            color: Color(0xFF0F766E), size: 22),
                        SizedBox(width: 8),
                        Text(
                          'Pathology Summary & Assessment',
                          style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF0F172A)),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      pathologySummary,
                      style: const TextStyle(
                        fontSize: 14,
                        color: Color(0xFF334155),
                        height: 1.45,
                      ),
                    ),

                    // Dos and Don'ts lists
                    if (dos.isNotEmpty || donts.isNotEmpty) ...[
                      const SizedBox(height: 16),
                      const Divider(),
                      const SizedBox(height: 16),
                      const Row(
                        children: [
                          Icon(Icons.checklist_rtl_rounded,
                              color: Color(0xFF0F766E), size: 22),
                          SizedBox(width: 8),
                          Text(
                            'Post-Operative Recovery Guidelines',
                            style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF0F172A)),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      if (dos.isNotEmpty) ...[
                        const Text(
                          'CLINICAL PROTOCOLS (DOS):',
                          style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF059669),
                              letterSpacing: 0.5),
                        ),
                        const SizedBox(height: 6),
                        ...dos.map((item) => _buildGuidelineRow(
                            Icons.check_circle_outline_rounded,
                            const Color(0xFF059669),
                            item.toString())),
                        const SizedBox(height: 16),
                      ],
                      if (donts.isNotEmpty) ...[
                        const Text(
                          'CONTRAINDICATED ACTIONS (DON\'TS):',
                          style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFFDC2626),
                              letterSpacing: 0.5),
                        ),
                        const SizedBox(height: 6),
                        ...donts.map((item) => _buildGuidelineRow(
                            Icons.cancel_outlined,
                            const Color(0xFFDC2626),
                            item.toString())),
                      ],
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildProfileRow(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20, color: const Color(0xFF0F766E)),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                    fontSize: 11,
                    color: Color(0xFF94A3B8),
                    fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 2),
              Text(
                value,
                style: const TextStyle(
                    fontSize: 14.5,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF334155)),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLegendDot(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(
          label,
          style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
              color: Color(0xFF475569)),
        ),
      ],
    );
  }

  Widget _buildGuidelineRow(IconData icon, Color color, String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 18, color: color),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                  fontSize: 13.5, height: 1.35, color: Color(0xFF334155)),
            ),
          ),
        ],
      ),
    );
  }
}
