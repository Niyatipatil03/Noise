import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:path_provider/path_provider.dart';
import '../models/detection_result.dart';

class ResultsScreen extends StatelessWidget {
  final VehicleSession session;
  final List<DetectionResult> results;

  const ResultsScreen({super.key, required this.session, required this.results});

  @override
  Widget build(BuildContext context) {
    final notOk  = session.isNotOk;
    final color  = notOk ? const Color(0xFFE53935) : const Color(0xFF43A047);
    final breakdown = session.noiseBreakdown;

    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0A),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.white),
        title: const Text('Session Report',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_rounded, color: Colors.white70),
            onPressed: () => _saveReport(context),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(children: [

          // Verdict banner
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [color, color.withOpacity(0.5)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [BoxShadow(color: color.withOpacity(0.4),
                  blurRadius: 20, offset: const Offset(0, 8))],
            ),
            child: Column(children: [
              Icon(notOk ? Icons.warning_rounded : Icons.verified_rounded,
                  color: Colors.white, size: 48),
              const SizedBox(height: 8),
              Text(notOk ? 'NOT OK' : 'OK',
                  style: const TextStyle(color: Colors.white,
                      fontSize: 32, fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              Text(session.vehicleName,
                  style: const TextStyle(color: Colors.white70, fontSize: 16)),
            ]),
          ).animate().scale(duration: 400.ms, curve: Curves.elasticOut),

          const SizedBox(height: 20),

          // Stats grid
          Row(children: [
            _statTile('Total', '${session.totalWindows}', Icons.grid_view, Colors.blueGrey),
            const SizedBox(width: 10),
            _statTile('OK', '${session.okCount}', Icons.check_circle, const Color(0xFF43A047)),
            const SizedBox(width: 10),
            _statTile('NOT OK', '${session.notOkCount}', Icons.cancel, const Color(0xFFE53935)),
          ]).animate().slideY(begin: 0.3, duration: 400.ms, delay: 100.ms).fadeIn(),

          const SizedBox(height: 16),

          // NOT OK Rate bar
          _card(
            Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Text('NOT OK Rate', style: TextStyle(color: Colors.white70, fontSize: 13)),
              const SizedBox(height: 8),
              Row(children: [
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: LinearProgressIndicator(
                      value: session.notOkRate,
                      minHeight: 12,
                      backgroundColor: Colors.white12,
                      valueColor: AlwaysStoppedAnimation<Color>(color),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Text('${(session.notOkRate * 100).toStringAsFixed(1)}%',
                    style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 16)),
              ]),
            ]),
          ).animate().slideY(begin: 0.3, duration: 400.ms, delay: 200.ms).fadeIn(),

          const SizedBox(height: 16),

          // Noise breakdown
          if (breakdown.isNotEmpty)
            _card(
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Row(children: [
                  Icon(Icons.equalizer_rounded, color: Colors.orange, size: 18),
                  SizedBox(width: 8),
                  Text('Noise Type Breakdown',
                      style: TextStyle(color: Colors.white,
                          fontWeight: FontWeight.bold, fontSize: 15)),
                ]),
                const SizedBox(height: 14),
                ...breakdown.entries.map((e) {
                  final pct = session.notOkCount > 0 ? e.value / session.notOkCount : 0.0;
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(children: [
                          Container(width: 10, height: 10,
                              decoration: const BoxDecoration(
                                  color: Colors.orange, shape: BoxShape.circle)),
                          const SizedBox(width: 8),
                          Text(e.key, style: const TextStyle(color: Colors.white70, fontSize: 13)),
                          const Spacer(),
                          Text('${e.value} windows',
                              style: const TextStyle(color: Colors.white,
                                  fontWeight: FontWeight.bold, fontSize: 13)),
                        ]),
                        const SizedBox(height: 6),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: LinearProgressIndicator(
                            value: pct.toDouble(),
                            minHeight: 6,
                            backgroundColor: Colors.white12,
                            color: Colors.orange,
                          ),
                        ),
                      ],
                    ),
                  );
                }),
              ]),
            ).animate().slideY(begin: 0.3, duration: 400.ms, delay: 300.ms).fadeIn(),

          const SizedBox(height: 16),

          // Time info
          _card(Column(children: [
            _infoRow(Icons.play_circle_outline, 'Start',
                _fmt(session.startTime)),
            if (session.endTime != null)
              _infoRow(Icons.stop_circle_outlined, 'End',
                  _fmt(session.endTime!)),
            if (session.endTime != null)
              _infoRow(Icons.timer_outlined, 'Duration',
                  _duration(session.startTime, session.endTime!)),
          ])).animate().slideY(begin: 0.3, duration: 400.ms, delay: 400.ms).fadeIn(),

          const SizedBox(height: 24),

          // Save button
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF1976D2),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
              icon: const Icon(Icons.save_rounded, color: Colors.white),
              label: const Text('SAVE REPORT',
                  style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15)),
              onPressed: () => _saveReport(context),
            ),
          ).animate().slideY(begin: 0.3, duration: 400.ms, delay: 500.ms).fadeIn(),
          const SizedBox(height: 12),
        ]),
      ),
    );
  }

  Widget _card(Widget child) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: const Color(0xFF1A1A1A),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: Colors.white.withOpacity(0.06)),
    ),
    child: child,
  );

  Widget _statTile(String label, String value, IconData icon, Color color) => Expanded(
    child: Container(
      padding: const EdgeInsets.symmetric(vertical: 16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(children: [
        Icon(icon, color: color, size: 22),
        const SizedBox(height: 6),
        Text(value, style: TextStyle(color: color, fontSize: 22, fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(color: Colors.white38, fontSize: 11)),
      ]),
    ),
  );

  Widget _infoRow(IconData icon, String label, String value) => Padding(
    padding: const EdgeInsets.only(bottom: 8),
    child: Row(children: [
      Icon(icon, color: Colors.white38, size: 16),
      const SizedBox(width: 8),
      Text(label, style: const TextStyle(color: Colors.white38, fontSize: 13)),
      const Spacer(),
      Text(value, style: const TextStyle(color: Colors.white70, fontSize: 13)),
    ]),
  );

  String _fmt(DateTime dt) =>
      '${dt.hour.toString().padLeft(2,'0')}:${dt.minute.toString().padLeft(2,'0')}  '
      '${dt.day}/${dt.month}/${dt.year}';

  String _duration(DateTime s, DateTime e) {
    final diff = e.difference(s);
    final m = diff.inMinutes, sec = diff.inSeconds % 60;
    return '${m}m ${sec}s';
  }

  Future<void> _saveReport(BuildContext context) async {
    try {
      final dir  = await getExternalStorageDirectory() ??
                   await getApplicationDocumentsDirectory();
      final ts   = DateTime.now().toString().replaceAll(':', '-').split('.').first;
      final file = File('${dir.path}/BSR_${session.vehicleName}_$ts.txt');
      await file.writeAsString(_reportText());
      if (context.mounted) _snack(context, 'Saved: ${file.path}');
    } catch (e) {
      if (context.mounted) _snack(context, 'Save failed: $e');
    }
  }

  void _snack(BuildContext ctx, String msg) =>
      ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(content: Text(msg)));

  String _reportText() {
    final sb = StringBuffer();
    sb.writeln('═══════════════════════════════════════');
    sb.writeln('  BSR Noise Detection Report');
    sb.writeln('  Vehicle  : ${session.vehicleName}');
    sb.writeln('  Date     : ${_fmt(session.startTime)}');
    sb.writeln('═══════════════════════════════════════');
    sb.writeln('Overall  : ${session.isNotOk ? "NOT OK" : "OK"}');
    sb.writeln('Windows  : ${session.totalWindows}');
    sb.writeln('OK       : ${session.okCount}');
    sb.writeln('NOT OK   : ${session.notOkCount}');
    sb.writeln('Rate     : ${(session.notOkRate * 100).toStringAsFixed(1)}%');
    if (session.noiseBreakdown.isNotEmpty) {
      sb.writeln('\nNoise Breakdown:');
      session.noiseBreakdown.forEach((k, v) => sb.writeln('  • $k : $v windows'));
    }
    sb.writeln('\nGenerated by BSR Noise Detector v1.0');
    sb.writeln('═══════════════════════════════════════');
    return sb.toString();
  }
}
