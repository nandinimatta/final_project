import 'dart:io';
import 'package:flutter/material.dart';

import 'screens/splash_screen.dart';

class MyHttpOverrides extends HttpOverrides {
  @override
  HttpClient createHttpClient(SecurityContext? context) {
    return super.createHttpClient(context)
      ..badCertificateCallback = (X509Certificate cert, String host, int port) => true;
  }
}

void main() {
  HttpOverrides.global = MyHttpOverrides();
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
