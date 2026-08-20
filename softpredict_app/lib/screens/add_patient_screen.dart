import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';

class AddPatientScreen extends StatefulWidget {
  final String? doctorName;
  const AddPatientScreen({super.key, this.doctorName});

  @override
  State<AddPatientScreen> createState() => _AddPatientScreenState();
}

class _AddPatientScreenState extends State<AddPatientScreen> {
  final _formKey = GlobalKey<FormState>();
  final ApiService _apiService = ApiService();
  final ImagePicker _imagePicker = ImagePicker();

  // Controllers
  final _idController = TextEditingController();
  final _nameController = TextEditingController();
  final _dobController = TextEditingController();
  final _ageController = TextEditingController();
  final _problemController = TextEditingController();

  // Dropdown States
  String _selectedGender = 'Male';
  String _selectedTreatmentMethod = 'teeth'; // teeth or jaw
  
  // Image States
  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  bool _isSaving = false;
  DateTime? _selectedDate;

  Future<void> _selectDate(BuildContext context) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: _selectedDate ?? DateTime.now().subtract(const Duration(days: 365 * 25)),
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.light(
              primary: Color(0xFF0F766E),
              onPrimary: Colors.white,
              onSurface: Color(0xFF0F172A),
            ),
          ),
          child: child!,
        );
      },
    );

    if (picked != null) {
      setState(() {
        _selectedDate = picked;
        _dobController.text = "${picked.year}-${picked.month.toString().padLeft(2, '0')}-${picked.day.toString().padLeft(2, '0')}";
        
        // Calculate Age
        final today = DateTime.now();
        int age = today.year - picked.year;
        if (today.month < picked.month || (today.month == picked.month && today.day < picked.day)) {
          age--;
        }
        _ageController.text = age.toString();
      });
    }
  }

  Future<void> _pickImage() async {
    try {
      final picked = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 90,
      );
      if (picked == null) return;

      final bytes = await picked.readAsBytes();
      setState(() {
        _selectedImageBytes = bytes;
        _selectedImageName = picked.name;
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error selecting image: $e')),
      );
    }
  }

  Future<void> _saveRecord() async {
    if (!_formKey.currentState!.validate()) return;

    if (_selectedImageBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select a clinical portrait photograph.'),
          backgroundColor: Color(0xFFDC2626),
        ),
      );
      return;
    }

    setState(() {
      _isSaving = true;
    });

    try {
      await _apiService.createRecord(
        bytes: _selectedImageBytes!,
        filename: _selectedImageName ?? 'patient.jpg',
        patientId: _idController.text.trim(),
        name: _nameController.text.trim(),
        age: int.parse(_ageController.text),
        dob: _dobController.text.trim(),
        gender: _selectedGender,
        problem: _problemController.text.trim(),
        treatmentMethod: _selectedTreatmentMethod,
        doctorUsername: widget.doctorName,
      );

      if (!mounted) return;
      setState(() {
        _isSaving = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Patient record saved and analyzed successfully.'),
          backgroundColor: Color(0xFF059669),
        ),
      );

      Navigator.of(context).pop(true); // Return true to trigger refresh
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isSaving = false;
      });
      showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Clinical Processing Error'),
          content: Text(e.toString().replaceAll('Exception:', '').trim()),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK', style: TextStyle(color: Color(0xFF0F766E))),
            ),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0F766E),
        foregroundColor: Colors.white,
        title: const Text(
          'Register New Patient',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      body: _isSaving
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(color: Color(0xFF0F766E)),
                  SizedBox(height: 20),
                  Text(
                    'Analyzing Facial Structures...',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF1E293B)),
                  ),
                  SizedBox(height: 6),
                  Text(
                    'Extracting coordinates and running prediction model',
                    style: TextStyle(fontSize: 13, color: Color(0xFF64748B)),
                  ),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20.0),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Section 1: Core Patient Profile
                    _buildSectionHeader('1. Core Patient Profile'),
                    const SizedBox(height: 12),
                    _buildProfileFields(),
                    const SizedBox(height: 24),

                    // Section 2: Clinical Details
                    _buildSectionHeader('2. Diagnosis & Treatment Target'),
                    const SizedBox(height: 12),
                    _buildClinicalFields(),
                    const SizedBox(height: 24),

                    // Section 3: Clinical Photograph
                    _buildSectionHeader('3. Clinical Face Photograph'),
                    const SizedBox(height: 12),
                    _buildPhotoSelector(),
                    const SizedBox(height: 36),

                    // Submit Buttons
                    SizedBox(
                      height: 52,
                      child: FilledButton.icon(
                        onPressed: _saveRecord,
                        style: FilledButton.styleFrom(
                          backgroundColor: const Color(0xFF0F766E),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        icon: const Icon(Icons.check_circle_outline_rounded),
                        label: const Text(
                          'Save Record & Run Prediction',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),
                    const SizedBox(height: 24),
                  ],
                ),
              ),
            ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 16,
        fontWeight: FontWeight.bold,
        color: Color(0xFF0F172A),
      ),
    );
  }

  Widget _buildProfileFields() {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: Color(0xFFE2E8F0)),
      ),
      color: Colors.white,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Patient ID Number / MRN
            TextFormField(
              controller: _idController,
              decoration: _inputDecoration('Hospital ID Number / MRN', Icons.badge_outlined),
              validator: (value) {
                if (value == null || value.trim().isEmpty) {
                  return 'Enter patient Medical Record Number (MRN)';
                }
                return null;
              },
            ),
            const SizedBox(height: 16),

            // Legal Name
            TextFormField(
              controller: _nameController,
              decoration: _inputDecoration('Full Legal Name', Icons.person_outline_rounded),
              validator: (value) {
                if (value == null || value.trim().isEmpty) {
                  return 'Enter patient full legal name';
                }
                return null;
              },
            ),
            const SizedBox(height: 16),

            // DOB & Age Row
            Row(
              children: [
                Expanded(
                  flex: 2,
                  child: TextFormField(
                    controller: _dobController,
                    readOnly: true,
                    onTap: () => _selectDate(context),
                    decoration: _inputDecoration('Date of Birth', Icons.calendar_today_outlined).copyWith(
                      hintText: 'YYYY-MM-DD',
                    ),
                    validator: (value) {
                      if (value == null || value.trim().isEmpty) {
                        return 'Select DOB';
                      }
                      return null;
                    },
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  flex: 1,
                  child: TextFormField(
                    controller: _ageController,
                    keyboardType: TextInputType.number,
                    decoration: _inputDecoration('Age', null).copyWith(
                      counterText: '',
                    ),
                    maxLength: 3,
                    validator: (value) {
                      if (value == null || value.trim().isEmpty) {
                        return 'Required';
                      }
                      if (int.tryParse(value) == null) {
                        return 'Invalid';
                      }
                      return null;
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Gender Identity Dropdown
            DropdownButtonFormField<String>(
              initialValue: _selectedGender,
              decoration: _inputDecoration('Biological Sex & Gender Identity', Icons.wc_outlined),
              items: <String>['Male', 'Female', 'Transgender', 'Non-binary', 'Other']
                  .map<DropdownMenuItem<String>>((String value) {
                return DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                );
              }).toList(),
              onChanged: (String? newValue) {
                if (newValue != null) {
                  setState(() {
                    _selectedGender = newValue;
                  });
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildClinicalFields() {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: Color(0xFFE2E8F0)),
      ),
      color: Colors.white,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Problem Description
            TextFormField(
              controller: _problemController,
              maxLines: 3,
              decoration: _inputDecoration(
                'Primary Injury / Pathology',
                Icons.medical_services_outlined,
              ).copyWith(
                hintText: 'e.g., 3cm jagged laceration to the right cheek, jaw asymmetry, severe dental misalignment...',
              ),
              validator: (value) {
                if (value == null || value.trim().isEmpty) {
                  return 'Describe the patient primary clinical issue';
                }
                return null;
              },
            ),
            const SizedBox(height: 16),

            // Modality Dropdown
            DropdownButtonFormField<String>(
              initialValue: _selectedTreatmentMethod,
              decoration: _inputDecoration('Orthopedic Refinement Modality', Icons.settings_accessibility_rounded),
              items: const [
                DropdownMenuItem(
                  value: 'teeth',
                  child: Text('Orthodontic Teeth / Bite Correction'),
                ),
                DropdownMenuItem(
                  value: 'jaw',
                  child: Text('Orthognathic Jaw Profile Correction'),
                ),
              ],
              onChanged: (String? newValue) {
                if (newValue != null) {
                  setState(() {
                    _selectedTreatmentMethod = newValue;
                  });
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPhotoSelector() {
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: Color(0xFFE2E8F0)),
      ),
      color: Colors.white,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: _pickImage,
        child: Container(
          height: 180,
          alignment: Alignment.center,
          child: _selectedImageBytes == null
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.add_a_photo_outlined, size: 48, color: Colors.grey[400]),
                    const SizedBox(height: 12),
                    const Text(
                      'Select Patient Portrait Photograph',
                      style: TextStyle(fontWeight: FontWeight.bold, color: Color(0xFF475569)),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      'Ensure frontal face alignment with neutral expression',
                      style: TextStyle(fontSize: 12, color: Color(0xFF94A3B8)),
                    ),
                  ],
                )
              : Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.memory(_selectedImageBytes!, fit: BoxFit.cover),
                    Container(
                      color: Colors.black38,
                      child: const Center(
                        child: Icon(Icons.cached_rounded, color: Colors.white, size: 36),
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }

  InputDecoration _inputDecoration(String labelText, IconData? prefixIcon) {
    return InputDecoration(
      labelText: labelText,
      prefixIcon: prefixIcon != null ? Icon(prefixIcon, color: const Color(0xFF0F766E)) : null,
      labelStyle: const TextStyle(fontSize: 14, color: Color(0xFF64748B)),
      contentPadding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFFCBD5E1)),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFFCBD5E1)),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFF0F766E), width: 1.5),
      ),
    );
  }
}
