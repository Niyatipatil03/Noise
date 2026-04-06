import 'dart:async';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import '../models/detection_result.dart';
import '../services/audio_recorder_service.dart';
import '../services/noise_classifier.dart';
import 'results_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _recorder   = AudioRecorderService();
  final _classifier = NoiseClassifier();
  final _history    = <DetectionResult>[];

  bool _isRecording  = false;
  bool _modelLoaded  = false;
  DetectionResult? _latest;
  StreamSubscription? _clipSub;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      await _classifier.init();
      setState(() => _modelLoaded = true);
    } catch (e) {
      _showSnack('Failed to load model: $e');
    }
  }

  Future<void> _startRecording() async {
    // Check permission
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      _showSnack('Microphone permission denied');
      return;
    }

    await _recorder.start();
    setState(() { _isRecording = true; _history.clear(); _latest = null; });

    _clipSub = _recorder.clips.listen((pcm) async {
      try {
        final result = await Future.microtask(() => _classifier.classify(pcm));
        if (mounted) setState(() { _latest = result; _history.add(result); });
      } catch (e) {
        debugPrint('Classification error: $e');
      }
    });
  }

  Future<void> _stopRecording() async {
    await _clipSub?.cancel();
    await _recorder.stop();
    setState(() => _isRecording = false);
  }

  void _showSnack(String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));

  @override
  void dispose() {
    _clipSub?.cancel();
    _recorder.dispose();
    _classifier.dispose();
    super.dispose();
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E1E1E),
        title: const Text('BSR Noise Detector', style: TextStyle(color: Colors.white)),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _buildStatusCard(),
            const SizedBox(height: 16),
            _buildStatsRow(),
            const Spacer(),
            if (!_modelLoaded)
              const Text('Loading model…', style: TextStyle(color: Colors.white54)),
            const SizedBox(height: 12),
            _buildStartStopButton(),
            const SizedBox(height: 12),
            if (_history.isNotEmpty && !_isRecording)
              _buildViewResultsButton(),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusCard() {
    Color cardColor;
    String statusText;
    String subText;

    if (_latest == null) {
      cardColor  = const Color(0xFF455A64);
      statusText = _isRecording ? 'Listening…' : 'Tap START to begin';
      subText    = '';
    } else if (_latest!.isNotOk) {
      cardColor  = const Color(0xFFC62828);
      statusText = '⚠  NOT OK';
      subText    = 'Noise: ${_latest!.noiseTypeName}  •  '
                   '${(_latest!.notOkProbability * 100).toStringAsFixed(1)}% confidence';
    } else {
      cardColor  = const Color(0xFF2E7D32);
      statusText = '✓  OK';
      subText    = '${(_latest!.okProbability * 100).toStringAsFixed(1)}% confident';
    }

    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      width: double.infinity,
      height: 140,
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [BoxShadow(color: cardColor.withOpacity(0.4), blurRadius: 12, offset: const Offset(0,4))],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(statusText, style: const TextStyle(color: Colors.white, fontSize: 26, fontWeight: FontWeight.bold)),
          if (subText.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(subText, style: const TextStyle(color: Colors.white70, fontSize: 14)),
          ],
        ],
      ),
    );
  }

  Widget _buildStatsRow() {
    final total  = _history.length;
    final notOk  = _history.where((r) => r.isNotOk).length;
    final ok     = total - notOk;
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _statChip('Windows', '$total', Colors.blueGrey),
        _statChip('OK', '$ok', const Color(0xFF2E7D32)),
        _statChip('NOT OK', '$notOk', const Color(0xFFC62828)),
      ],
    );
  }

  Widget _statChip(String label, String value, Color color) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
    decoration: BoxDecoration(color: color.withOpacity(0.2), borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.5))),
    child: Column(children: [
      Text(value, style: TextStyle(color: color, fontSize: 20, fontWeight: FontWeight.bold)),
      Text(label,  style: const TextStyle(color: Colors.white54, fontSize: 12)),
    ]),
  );

  Widget _buildStartStopButton() => SizedBox(
    width: double.infinity,
    height: 56,
    child: ElevatedButton(
      style: ElevatedButton.styleFrom(
        backgroundColor: _isRecording ? const Color(0xFFC62828) : const Color(0xFF1976D2),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      ),
      onPressed: _modelLoaded
          ? (_isRecording ? _stopRecording : _startRecording)
          : null,
      child: Text(
        _isRecording ? 'STOP' : 'START ANALYSIS',
        style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
      ),
    ),
  );

  Widget _buildViewResultsButton() => SizedBox(
    width: double.infinity,
    height: 48,
    child: OutlinedButton(
      style: OutlinedButton.styleFrom(
        side: const BorderSide(color: Color(0xFF1976D2)),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      ),
      onPressed: () => Navigator.push(context,
          MaterialPageRoute(builder: (_) => ResultsScreen(history: _history))),
      child: const Text('VIEW SESSION REPORT', style: TextStyle(color: Color(0xFF1976D2))),
    ),
  );
}
