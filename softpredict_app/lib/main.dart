import 'package:flutter/material.dart';

import 'screens/splash_screen.dart';

void main() {
  runApp(const SoftPredictApp());
}

class SoftPredictApp extends StatelessWidget {
  const SoftPredictApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'SoftPredict',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0F766E)),
        scaffoldBackgroundColor: const Color(0xFFF5F7FA),
        useMaterial3: true,
      ),
      home: const SplashScreen(),
    );
  }
}
