import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import '../models/detection_result.dart';

class ResultsScreen extends StatelessWidget {
  final List<DetectionResult> history;
  const ResultsScreen({super.key, required this.history});

  int get _total  => history.length;
  int get _notOk  => history.where((r) => r.isNotOk).length;
  int get _ok     => _total - _notOk;
  double get _notOkPct => _total > 0 ? _notOk / _total : 0;

  Map<String, int> get _typeBreakdown {
    final map = <String, int>{};
    for (final r in history.where((r) => r.isNotOk)) {
      map[r.noiseTypeName] = (map[r.noiseTypeName] ?? 0) + 1;
    }
    final sorted = Map.fromEntries(map.entries.toList()..sort((a, b) => b.value.compareTo(a.value)));
    return sorted;
  }

  @override
  Widget build(BuildContext context) {
    final isNotOk = _notOk > _ok;
    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E1E1E),
        title: const Text('Session Report', style: TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // Overall verdict
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: isNotOk ? const Color(0xFFC62828) : const Color(0xFF2E7D32),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                isNotOk ? '⚠  NOT OK' : '✓  OK',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 16),

            // Stats card
            _card(Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _statLine('Total windows analysed', '$_total'),
                _statLine('OK windows',     '$_ok',    color: const Color(0xFF4CAF50)),
                _statLine('NOT OK windows', '$_notOk', color: const Color(0xFFF44336)),
                _statLine('NOT OK rate',    '${(_notOkPct * 100).toStringAsFixed(1)}%'),
                const SizedBox(height: 8),
                LinearProgressIndicator(
                  value: _notOkPct,
                  backgroundColor: Colors.white12,
                  color: const Color(0xFFF44336),
                  minHeight: 10,
                ),
              ],
            )),
            const SizedBox(height: 16),

            // Noise breakdown
            _card(Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Noise Type Breakdown',
                    style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
                const SizedBox(height: 12),
                if (_typeBreakdown.isEmpty)
                  const Text('No noise detected', style: TextStyle(color: Colors.white54))
                else
                  ..._typeBreakdown.entries.map((e) => Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Row(children: [
                      const Text('• ', style: TextStyle(color: Colors.orange)),
                      Text('${e.key}', style: const TextStyle(color: Colors.white70)),
                      const Spacer(),
                      Text('${e.value} windows',
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                    ]),
                  )),
              ],
            )),
            const SizedBox(height: 24),

            // Save button
            SizedBox(
              width: double.infinity,
              height: 48,
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1976D2),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                ),
                onPressed: () => _saveReport(context),
                child: const Text('SAVE REPORT', style: TextStyle(color: Colors.white)),
              ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              height: 48,
              child: OutlinedButton(
                style: OutlinedButton.styleFrom(
                  side: const BorderSide(color: Colors.white24),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                ),
                onPressed: () => Navigator.pop(context),
                child: const Text('BACK', style: TextStyle(color: Colors.white70)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _card(Widget child) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: const Color(0xFF1E1E1E),
      borderRadius: BorderRadius.circular(12),
    ),
    child: child,
  );

  Widget _statLine(String label, String value, {Color? color}) => Padding(
    padding: const EdgeInsets.only(bottom: 6),
    child: Row(children: [
      Text(label, style: const TextStyle(color: Colors.white54, fontFamily: 'monospace')),
      const Spacer(),
      Text(value, style: TextStyle(color: color ?? Colors.white,
          fontFamily: 'monospace', fontWeight: FontWeight.bold)),
    ]),
  );

  Future<void> _saveReport(BuildContext context) async {
    try {
      final dir  = await getExternalStorageDirectory() ?? await getApplicationDocumentsDirectory();
      final ts   = DateTime.now().toString().replaceAll(':', '-').split('.').first;
      final file = File('${dir.path}/BSR_Report_$ts.txt');
      await file.writeAsString(_buildReportText());
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Report saved: ${file.path}')));
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Save failed: $e')));
      }
    }
  }

  String _buildReportText() {
    final sb = StringBuffer();
    sb.writeln('═══════════════════════════════════════');
    sb.writeln('  BSR Noise Detection Report');
    sb.writeln('  ${DateTime.now()}');
    sb.writeln('═══════════════════════════════════════');
    sb.writeln('Overall  : ${_notOk > _ok ? "NOT OK" : "OK"}');
    sb.writeln('Total    : $_total windows');
    sb.writeln('OK       : $_ok windows');
    sb.writeln('NOT OK   : $_notOk windows');
    sb.writeln('Rate     : ${(_notOkPct * 100).toStringAsFixed(1)}%');
    if (_typeBreakdown.isNotEmpty) {
      sb.writeln('\nNoise Breakdown:');
      _typeBreakdown.forEach((k, v) => sb.writeln('  • $k : $v windows'));
    }
    sb.writeln('\nGenerated by BSR Noise Detector v1.0');
    sb.writeln('═══════════════════════════════════════');
    return sb.toString();
  }
}
