import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:permission_handler/permission_handler.dart';
import '../models/detection_result.dart';
import '../services/audio_recorder_service.dart';
import '../services/noise_classifier.dart';
import '../services/history_service.dart';
import '../widgets/record_button.dart';
import 'results_screen.dart';
import 'history_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _recorder    = AudioRecorderService();
  final _classifier  = NoiseClassifier();
  final _histSvc     = HistoryService();
  final _results     = <DetectionResult>[];

  bool _isRecording  = false;
  bool _modelLoaded  = false;
  String? _modelError;
  DetectionResult? _latest;
  StreamSubscription? _clipSub;
  VehicleSession? _currentSession;
  String _vehicleName = '';

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      await _classifier.init();
      if (mounted) setState(() => _modelLoaded = true);
    } catch (e) {
      if (mounted) setState(() => _modelError = e.toString());
    }
  }

  // ── Recording flow ────────────────────────────────────────────────────────

  Future<void> _onRecordTap() async {
    if (_isRecording) { _stopRecording(); return; }

    // Ask vehicle name
    final name = await _showVehicleDialog();
    if (name == null || name.trim().isEmpty) return;
    _vehicleName = name.trim();

    final ok = await Permission.microphone.request();
    if (!ok.isGranted) { _snack('Microphone permission denied'); return; }

    _results.clear();
    _currentSession = VehicleSession(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      vehicleName: _vehicleName,
      startTime: DateTime.now(),
      results: _results,
    );

    await _recorder.start();
    setState(() { _isRecording = true; _latest = null; });

    _clipSub = _recorder.clips.listen((pcm) async {
      try {
        final result = await Future.microtask(() => _classifier.classify(pcm));
        if (mounted) setState(() { _latest = result; _results.add(result); });
      } catch (e) { debugPrint('Classify error: $e'); }
    });
  }

  Future<void> _stopRecording() async {
    await _clipSub?.cancel();
    await _recorder.stop();

    if (_currentSession != null) {
      _currentSession!.endTime = DateTime.now();
      await _histSvc.save(_currentSession!);
    }
    setState(() => _isRecording = false);

    if (_results.isNotEmpty && mounted) {
      Navigator.push(context, _slideRoute(
        ResultsScreen(session: _currentSession!, results: List.from(_results)),
      ));
    }
  }

  // ── Dialogs ───────────────────────────────────────────────────────────────

  Future<String?> _showVehicleDialog() => showDialog<String>(
    context: context,
    builder: (ctx) {
      String name = '';
      return AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Vehicle ID / Name',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        content: TextField(
          autofocus: true,
          style: const TextStyle(color: Colors.white),
          decoration: InputDecoration(
            hintText: 'e.g. VIN-12345 or Test Car 1',
            hintStyle: const TextStyle(color: Colors.white38),
            filled: true,
            fillColor: const Color(0xFF2C2C2C),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(10),
                borderSide: BorderSide.none),
            prefixIcon: const Icon(Icons.directions_car, color: Color(0xFF1976D2)),
          ),
          onChanged: (v) => name = v,
          onSubmitted: (_) => Navigator.pop(ctx, name),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF1976D2),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8))),
            onPressed: () => Navigator.pop(ctx, name),
            child: const Text('START', style: TextStyle(color: Colors.white)),
          ),
        ],
      );
    },
  );

  void _snack(String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));

  PageRoute _slideRoute(Widget page) => PageRouteBuilder(
    pageBuilder: (_, a, __) => page,
    transitionsBuilder: (_, a, __, child) =>
        SlideTransition(position: Tween(begin: const Offset(1, 0), end: Offset.zero)
            .animate(CurvedAnimation(parent: a, curve: Curves.easeInOut)), child: child),
  );

  @override
  void dispose() {
    _clipSub?.cancel();
    _recorder.dispose();
    _classifier.dispose();
    super.dispose();
  }

  // ── UI ────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0A),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Row(children: [
          Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              color: const Color(0xFF1976D2).withOpacity(0.2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(Icons.car_repair, color: Color(0xFF1976D2), size: 20),
          ),
          const SizedBox(width: 10),
          const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('BSR Detector', style: TextStyle(color: Colors.white,
                  fontSize: 16, fontWeight: FontWeight.bold)),
              Text('Vehicle Noise Analysis', style: TextStyle(
                  color: Colors.white38, fontSize: 11)),
            ],
          ),
        ]),
        actions: [
          IconButton(
            icon: const Icon(Icons.history_rounded, color: Colors.white70),
            tooltip: 'Session History',
            onPressed: () => Navigator.push(context,
                _slideRoute(const HistoryScreen())),
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Column(children: [
                const SizedBox(height: 8),
                if (_modelError != null) _buildModelErrorBanner(),
                _buildStatusCard(),
                const SizedBox(height: 16),
                _buildStatsRow(),
                const SizedBox(height: 16),
                if (_latest != null && _latest!.isNotOk) _buildNoiseTypeBar(),
                if (_isRecording) _buildWaveformIndicator(),
              ]),
            ),
          ),

          // Bottom recording section
          _buildBottomSection(),
        ],
      ),
    );
  }

  Widget _buildModelErrorBanner() => Container(
    margin: const EdgeInsets.only(bottom: 16),
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: const Color(0xFF4A0000),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: Colors.red.withOpacity(0.5)),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Row(children: [
          Icon(Icons.error_outline_rounded, color: Colors.red, size: 18),
          SizedBox(width: 8),
          Text('Model Load Failed',
              style: TextStyle(color: Colors.red,
                  fontWeight: FontWeight.bold, fontSize: 14)),
        ]),
        const SizedBox(height: 10),
        const Text(
          'The TFLite model could not be loaded. To fix this:',
          style: TextStyle(color: Colors.white70, fontSize: 13),
        ),
        const SizedBox(height: 8),
        const Text(
          '1.  Train:   python scripts/train.py\n'
          '2.  Export: python scripts/export_tflite.py\n'
          '3.  Copy models/bsr_detector.tflite\n'
          '         → flutter_app/assets/\n'
          '4.  Rebuild and reinstall the APK',
          style: TextStyle(color: Colors.white54, fontSize: 12,
              fontFamily: 'monospace'),
        ),
        const SizedBox(height: 10),
        Text(
          _modelError ?? '',
          style: const TextStyle(color: Colors.red, fontSize: 11),
          maxLines: 3,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    ),
  ).animate().fadeIn(duration: 300.ms);

  Widget _buildStatusCard() {
    final Color cardColor;
    final String title;
    final String subtitle;
    final IconData icon;

    if (_latest == null && !_isRecording) {
      cardColor = const Color(0xFF1A237E);
      title     = 'Ready to Analyse';
      subtitle  = 'Enter vehicle name and tap record';
      icon      = Icons.sensors;
    } else if (_latest == null) {
      cardColor = const Color(0xFF1B5E20);
      title     = 'Listening...';
      subtitle  = 'Analysing audio — drive the BSR track';
      icon      = Icons.hearing;
    } else if (_latest!.isNotOk) {
      cardColor = const Color(0xFF7F0000);
      title     = 'NOT OK  ⚠';
      subtitle  = _latest!.noiseTypeName;
      icon      = Icons.warning_rounded;
    } else {
      cardColor = const Color(0xFF1B5E20);
      title     = 'OK  ✓';
      subtitle  = '${(_latest!.okProbability * 100).toStringAsFixed(1)}% confident';
      icon      = Icons.check_circle_rounded;
    }

    return AnimatedContainer(
      duration: const Duration(milliseconds: 500),
      curve: Curves.easeInOut,
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [cardColor, cardColor.withOpacity(0.6)],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
        boxShadow: [BoxShadow(color: cardColor.withOpacity(0.5), blurRadius: 20, offset: const Offset(0,8))],
      ),
      padding: const EdgeInsets.all(20),
      child: Row(children: [
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.15),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: Colors.white, size: 28),
        ),
        const SizedBox(width: 16),
        Expanded(child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(color: Colors.white,
                fontSize: 22, fontWeight: FontWeight.bold)),
            const SizedBox(height: 4),
            Text(subtitle, style: TextStyle(color: Colors.white.withOpacity(0.8), fontSize: 13)),
            if (_isRecording && _vehicleName.isNotEmpty) ...[
              const SizedBox(height: 6),
              Row(children: [
                const Icon(Icons.directions_car, color: Colors.white54, size: 14),
                const SizedBox(width: 4),
                Text(_vehicleName, style: const TextStyle(color: Colors.white54, fontSize: 12)),
              ]),
            ],
          ],
        )),
        if (_latest != null)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.15),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(
              '${(_latest!.notOkProbability * 100).toStringAsFixed(0)}%',
              style: const TextStyle(color: Colors.white,
                  fontWeight: FontWeight.bold, fontSize: 16),
            ),
          ),
      ]),
    ).animate().fadeIn(duration: 300.ms);
  }

  Widget _buildStatsRow() => Row(
    children: [
      _statCard('Total', '${_results.length}', Icons.bar_chart_rounded, Colors.blueGrey),
      const SizedBox(width: 10),
      _statCard('OK', '${_results.where((r) => !r.isNotOk).length}',
          Icons.check_circle_outline, const Color(0xFF43A047)),
      const SizedBox(width: 10),
      _statCard('NOT OK', '${_results.where((r) => r.isNotOk).length}',
          Icons.warning_amber_rounded, const Color(0xFFE53935)),
    ],
  );

  Widget _statCard(String label, String value, IconData icon, Color color) => Expanded(
    child: AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(children: [
        Icon(icon, color: color, size: 22),
        const SizedBox(height: 6),
        Text(value, style: TextStyle(color: color, fontSize: 20, fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(color: Colors.white38, fontSize: 11)),
      ]),
    ),
  );

  Widget _buildNoiseTypeBar() {
    if (_latest == null) return const SizedBox.shrink();
    final probs = _latest!.noiseTypeProbs;
    final names = NoiseClassifier.noiseTypeNames;
    final top3  = List.generate(probs.length, (i) => i)
        ..sort((a, b) => probs[b].compareTo(probs[a]));

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.orange.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(children: [
            Icon(Icons.equalizer_rounded, color: Colors.orange, size: 18),
            SizedBox(width: 8),
            Text('Noise Analysis', style: TextStyle(color: Colors.orange,
                fontWeight: FontWeight.bold, fontSize: 14)),
          ]),
          const SizedBox(height: 12),
          ...top3.take(3).map((i) => Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Text(names[i], style: const TextStyle(color: Colors.white70, fontSize: 12)),
                  const Spacer(),
                  Text('${(probs[i] * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(color: Colors.white, fontSize: 12,
                          fontWeight: FontWeight.bold)),
                ]),
                const SizedBox(height: 4),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: probs[i],
                    minHeight: 6,
                    backgroundColor: Colors.white12,
                    valueColor: AlwaysStoppedAnimation<Color>(
                        i == top3.first ? Colors.orange : Colors.orange.withOpacity(0.4)),
                  ),
                ),
              ],
            ),
          )),
        ],
      ),
    ).animate().slideX(begin: 0.2, duration: 300.ms).fadeIn();
  }

  Widget _buildWaveformIndicator() => Container(
    margin: const EdgeInsets.only(bottom: 16),
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: const Color(0xFF1A1A1A),
      borderRadius: BorderRadius.circular(14),
    ),
    child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
      const Icon(Icons.graphic_eq, color: Color(0xFF1976D2), size: 18),
      const SizedBox(width: 8),
      const Text('Recording in progress…',
          style: TextStyle(color: Colors.white54, fontSize: 13)),
    ]),
  ).animate(onPlay: (c) => c.repeat(reverse: true))
   .fadeIn(duration: 800.ms).then().fadeOut(duration: 800.ms);

  Widget _buildBottomSection() => Container(
    padding: const EdgeInsets.fromLTRB(20, 20, 20, 32),
    decoration: const BoxDecoration(
      color: Color(0xFF111111),
      borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
    ),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Handle bar
        Container(width: 40, height: 4,
            decoration: BoxDecoration(color: Colors.white24,
                borderRadius: BorderRadius.circular(2))),
        const SizedBox(height: 20),

        // Record button
        RecordButton(
          isRecording: _isRecording,
          enabled: _modelLoaded,
          onTap: _onRecordTap,
        ),
        const SizedBox(height: 16),

        Text(
          _isRecording
              ? 'Tap to stop recording'
              : _modelLoaded
                  ? 'Tap to start analysis'
                  : _modelError != null
                      ? 'Model unavailable — see banner above'
                      : 'Loading model…',
          style: const TextStyle(color: Colors.white54, fontSize: 13),
        ),

        if (_results.isNotEmpty && !_isRecording) ...[
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: Color(0xFF1976D2)),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
              icon: const Icon(Icons.assessment_rounded, color: Color(0xFF1976D2)),
              label: const Text('View Last Report',
                  style: TextStyle(color: Color(0xFF1976D2))),
              onPressed: () => Navigator.push(context, _slideRoute(
                ResultsScreen(session: _currentSession!,
                    results: List.from(_results)),
              )),
            ),
          ),
        ],
      ],
    ),
  );
}
