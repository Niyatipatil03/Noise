import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../models/detection_result.dart';
import '../services/history_service.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final _svc = HistoryService();
  List<VehicleSession> _sessions = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final sessions = await _svc.loadAll();
    if (mounted) setState(() { _sessions = sessions; _loading = false; });
  }

  Future<void> _clearAll() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: const Text('Clear History', style: TextStyle(color: Colors.white)),
        content: const Text('Delete all session records?',
            style: TextStyle(color: Colors.white70)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel')),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete All', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
    if (ok == true) {
      await _svc.deleteAll();
      setState(() => _sessions.clear());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0A),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.white),
        title: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Session History',
              style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          Text('${_sessions.length} sessions',
              style: const TextStyle(color: Colors.white38, fontSize: 11)),
        ]),
        actions: [
          if (_sessions.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_sweep_rounded, color: Colors.white54),
              onPressed: _clearAll,
            ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _sessions.isEmpty
              ? _buildEmpty()
              : _buildList(),
    );
  }

  Widget _buildEmpty() => Center(
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      Icon(Icons.history_rounded, size: 72, color: Colors.white12),
      const SizedBox(height: 16),
      const Text('No sessions yet', style: TextStyle(color: Colors.white38, fontSize: 16)),
      const SizedBox(height: 8),
      const Text('Complete a BSR analysis to see history',
          style: TextStyle(color: Colors.white24, fontSize: 13)),
    ]),
  );

  Widget _buildList() => ListView.builder(
    padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
    itemCount: _sessions.length,
    itemBuilder: (ctx, i) => _SessionCard(session: _sessions[i], index: i),
  );
}

class _SessionCard extends StatelessWidget {
  final VehicleSession session;
  final int index;
  const _SessionCard({required this.session, required this.index});

  @override
  Widget build(BuildContext context) {
    final isNotOk = session.isNotOk;
    final color   = isNotOk ? const Color(0xFFE53935) : const Color(0xFF43A047);
    final breakdown = session.noiseBreakdown;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: () => _showDetail(context),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Header row
                  Row(children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: color.withOpacity(0.15),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        isNotOk ? Icons.warning_rounded : Icons.check_circle_rounded,
                        color: color, size: 18,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(session.vehicleName,
                            style: const TextStyle(color: Colors.white,
                                fontWeight: FontWeight.bold, fontSize: 15)),
                        Text(_fmtDate(session.startTime),
                            style: const TextStyle(color: Colors.white38, fontSize: 11)),
                      ],
                    )),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: color.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: color.withOpacity(0.4)),
                      ),
                      child: Text(isNotOk ? 'NOT OK' : 'OK',
                          style: TextStyle(color: color,
                              fontWeight: FontWeight.bold, fontSize: 12)),
                    ),
                  ]),

                  const SizedBox(height: 12),

                  // Stats row
                  Row(children: [
                    _chip(Icons.grid_view, '${session.totalWindows} windows', Colors.blueGrey),
                    const SizedBox(width: 8),
                    _chip(Icons.warning_amber_rounded,
                        '${(session.notOkRate * 100).toStringAsFixed(0)}% NOT OK',
                        isNotOk ? Colors.red : Colors.green),
                  ]),

                  // Noise types
                  if (breakdown.isNotEmpty) ...[
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: breakdown.entries.take(3).map((e) => Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: Colors.orange.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.orange.withOpacity(0.3)),
                        ),
                        child: Text('${e.key}  ${e.value}×',
                            style: const TextStyle(color: Colors.orange, fontSize: 11)),
                      )).toList(),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    ).animate().slideX(begin: 0.15, duration: 300.ms,
        delay: Duration(milliseconds: index * 50)).fadeIn();
  }

  Widget _chip(IconData icon, String label, Color color) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Icon(icon, color: color, size: 13),
      const SizedBox(width: 4),
      Text(label, style: TextStyle(color: color, fontSize: 12)),
    ],
  );

  void _showDetail(BuildContext context) => showModalBottomSheet(
    context: context,
    backgroundColor: const Color(0xFF1A1A1A),
    shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
    builder: (_) => Padding(
      padding: const EdgeInsets.all(20),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Container(width: 40, height: 4,
            decoration: BoxDecoration(color: Colors.white24,
                borderRadius: BorderRadius.circular(2))),
        const SizedBox(height: 16),
        Text(session.vehicleName,
            style: const TextStyle(color: Colors.white,
                fontWeight: FontWeight.bold, fontSize: 18)),
        const SizedBox(height: 4),
        Text(_fmtDate(session.startTime),
            style: const TextStyle(color: Colors.white54)),
        const SizedBox(height: 16),
        const Divider(color: Colors.white12),
        _detailRow('Result',     session.isNotOk ? 'NOT OK ⚠' : 'OK ✓'),
        _detailRow('Windows',    '${session.totalWindows}'),
        _detailRow('OK',         '${session.okCount}'),
        _detailRow('NOT OK',     '${session.notOkCount}'),
        _detailRow('NOT OK Rate','${(session.notOkRate*100).toStringAsFixed(1)}%'),
        if (session.noiseBreakdown.isNotEmpty) ...[
          const Divider(color: Colors.white12),
          ...session.noiseBreakdown.entries.map((e) =>
              _detailRow('  • ${e.key}', '${e.value} windows')),
        ],
        const SizedBox(height: 20),
      ]),
    ),
  );

  Widget _detailRow(String label, String value) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 4),
    child: Row(children: [
      Text(label, style: const TextStyle(color: Colors.white54, fontSize: 13)),
      const Spacer(),
      Text(value, style: const TextStyle(color: Colors.white,
          fontWeight: FontWeight.w500, fontSize: 13)),
    ]),
  );

  String _fmtDate(DateTime dt) =>
      '${dt.day}/${dt.month}/${dt.year}  '
      '${dt.hour.toString().padLeft(2,'0')}:${dt.minute.toString().padLeft(2,'0')}';
}
