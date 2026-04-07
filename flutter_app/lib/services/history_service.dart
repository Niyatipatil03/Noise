import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/detection_result.dart';

class HistoryService {
  static const _key = 'bsr_session_history';

  Future<List<VehicleSession>> loadAll() async {
    final prefs = await SharedPreferences.getInstance();
    final raw   = prefs.getStringList(_key) ?? [];
    return raw.map((s) => VehicleSession.fromJson(jsonDecode(s))).toList()
      ..sort((a, b) => b.startTime.compareTo(a.startTime));
  }

  Future<void> save(VehicleSession session) async {
    final prefs    = await SharedPreferences.getInstance();
    final existing = prefs.getStringList(_key) ?? [];
    existing.insert(0, jsonEncode(session.toJson()));
    // Keep last 100 sessions
    if (existing.length > 100) existing.removeLast();
    await prefs.setStringList(_key, existing);
  }

  Future<void> deleteAll() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }
}
