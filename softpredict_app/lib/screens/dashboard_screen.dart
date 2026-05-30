import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'add_patient_screen.dart';
import 'patient_details_screen.dart';
import 'login_screen.dart';

class DashboardScreen extends StatefulWidget {
  final String doctorName;
  const DashboardScreen({super.key, required this.doctorName});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final ApiService _apiService = ApiService();
  List<dynamic> _allRecords = [];
  List<dynamic> _filteredRecords = [];
  bool _isLoading = true;
  String _searchQuery = '';
  final TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _fetchRecords();
  }

  Future<void> _fetchRecords() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final records = await _apiService.getRecords();
      if (!mounted) return;
      setState(() {
        _allRecords = records;
        _filterRecords(_searchQuery);
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load patient records: $e')),
      );
    }
  }

  void _filterRecords(String query) {
    setState(() {
      _searchQuery = query;
      if (query.isEmpty) {
        _filteredRecords = List.from(_allRecords);
      } else {
        final lowQuery = query.toLowerCase();
        _filteredRecords = _allRecords.where((record) {
          final name = (record['name'] ?? '').toString().toLowerCase();
          final id = (record['patient_id'] ?? '').toString().toLowerCase();
          final problem = (record['problem'] ?? '').toString().toLowerCase();
          return name.contains(lowQuery) || id.contains(lowQuery) || problem.contains(lowQuery);
        }).toList();
      }
    });
  }

  void _handleLogout() {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(builder: (_) => const LoginScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0F766E),
        foregroundColor: Colors.white,
        title: const Text(
          'Clinical Dashboard',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout_rounded),
            tooltip: 'Sign Out',
            onPressed: _handleLogout,
          ),
        ],
      ),
      body: Column(
        children: [
          // Doctor Profile Header Banner
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            color: const Color(0xFF0F172A),
            child: Row(
              children: [
                const CircleAvatar(
                  backgroundColor: Color(0xFF0F766E),
                  radius: 20,
                  child: Icon(Icons.person_pin_rounded, color: Colors.white, size: 24),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Welcome, Dr. ${widget.doctorName}',
                      style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15),
                    ),
                    const SizedBox(height: 2),
                    const Text(
                      'Department of Oral & Maxillofacial Surgery',
                      style: TextStyle(color: Colors.white60, fontSize: 11),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Search Bar
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: TextField(
              controller: _searchController,
              onChanged: _filterRecords,
              decoration: InputDecoration(
                hintText: 'Search by Patient Name, ID or Diagnosis...',
                prefixIcon: const Icon(Icons.search_rounded, color: Color(0xFF0F766E)),
                suffixIcon: _searchQuery.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear_rounded),
                        onPressed: () {
                          _searchController.clear();
                          _filterRecords('');
                        },
                      )
                    : null,
                filled: true,
                fillColor: Colors.white,
                contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 16),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Color(0xFFE2E8F0)),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Color(0xFFE2E8F0)),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Color(0xFF0F766E), width: 1.5),
                ),
              ),
            ),
          ),

          // Records Header
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 4),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Patient Record Database (${_filteredRecords.length})',
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF1E293B)),
                ),
                IconButton(
                  icon: const Icon(Icons.refresh_rounded, color: Color(0xFF0F766E)),
                  onPressed: _fetchRecords,
                  tooltip: 'Refresh Records',
                ),
              ],
            ),
          ),

          // Records List
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator(color: Color(0xFF0F766E)))
                : _filteredRecords.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.folder_open_outlined, size: 72, color: Colors.grey[300]),
                            const SizedBox(height: 16),
                            Text(
                              _searchQuery.isNotEmpty
                                  ? 'No matching patient records found'
                                  : 'No patient profiles registered yet',
                              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Color(0xFF64748B)),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _searchQuery.isNotEmpty
                                  ? 'Try refining your search text'
                                  : 'Click the "+" button to register a new profile',
                              style: const TextStyle(fontSize: 13, color: Color(0xFF94A3B8)),
                            ),
                          ],
                        ),
                      )
                    : RefreshIndicator(
                        onRefresh: _fetchRecords,
                        color: const Color(0xFF0F766E),
                        child: ListView.builder(
                          physics: const AlwaysScrollableScrollPhysics(),
                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                          itemCount: _filteredRecords.length,
                          itemBuilder: (context, index) {
                            final record = _filteredRecords[index];
                            return _buildRecordCard(record);
                          },
                        ),
                      ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: const Color(0xFF0F766E),
        foregroundColor: Colors.white,
        icon: const Icon(Icons.add),
        label: const Text('Add Patient'),
        onPressed: () async {
          final result = await Navigator.of(context).push<bool>(
            MaterialPageRoute(builder: (_) => const AddPatientScreen()),
          );
          if (result == true) {
            _fetchRecords();
          }
        },
      ),
    );
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

  Widget _buildRecordCard(dynamic record) {
    final patientId = record['patient_id'] ?? 'N/A';
    final name = record['name'] ?? 'Unknown';
    final problem = record['problem'] ?? 'No description';
    final treatmentMethod = (record['treatment_method'] ?? 'teeth').toString().toUpperCase();
    final age = record['age'] ?? 'N/A';
    final gender = record['gender'] ?? 'N/A';
    final beforeUrl = _resolveImageUrl(record['before_url'] ?? '');
    final afterUrl = _resolveImageUrl(record['after_url'] ?? '');

    return Card(
      elevation: 2,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      color: Colors.white,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () async {
          final result = await Navigator.of(context).push<bool>(
            MaterialPageRoute(
              builder: (_) => PatientDetailsScreen(record: record),
            ),
          );
          if (result == true) {
            _fetchRecords();
          }
        },
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Mini Preview Images Stack (Before -> After)
              Container(
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: const Color(0xFFF1F5F9),
                  border: Border.all(color: const Color(0xFFE2E8F0)),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: afterUrl.isNotEmpty
                      ? Image.network(
                          afterUrl,
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => const Icon(Icons.person, color: Color(0xFF0F766E)),
                        )
                      : beforeUrl.isNotEmpty
                          ? Image.network(
                              beforeUrl,
                              fit: BoxFit.cover,
                              errorBuilder: (_, __, ___) => const Icon(Icons.person, color: Color(0xFF0F766E)),
                            )
                          : const Icon(Icons.person, color: Color(0xFF0F766E), size: 36),
                ),
              ),
              const SizedBox(width: 16),

              // Patient Metadata
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            name,
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1E293B),
                            ),
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: treatmentMethod == 'TEETH' ? const Color(0xFFECFDF5) : const Color(0xFFEFF6FF),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            treatmentMethod,
                            style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w800,
                              color: treatmentMethod == 'TEETH' ? const Color(0xFF059669) : const Color(0xFF2563EB),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'MRN/ID: $patientId  •  $gender  •  Age: $age',
                      style: const TextStyle(
                        fontSize: 12.5,
                        color: Color(0xFF64748B),
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      problem,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 13,
                        color: Color(0xFF334155),
                        height: 1.3,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 4),
              const Center(
                child: Icon(Icons.chevron_right_rounded, color: Color(0xFF94A3B8)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
