class DetectionResult {
  final bool isNotOk;
  final double notOkProbability;
  final int noiseTypeIndex;
  final String noiseTypeName;
  final List<double> noiseTypeProbs;
  final DateTime timestamp;

  DetectionResult({
    required this.isNotOk,
    required this.notOkProbability,
    required this.noiseTypeIndex,
    required this.noiseTypeName,
    required this.noiseTypeProbs,
    required this.timestamp,
  });

  double get okProbability => 1.0 - notOkProbability;
}

class VehicleSession {
  final String id;
  final String vehicleName;
  final DateTime startTime;
  DateTime? endTime;
  final List<DetectionResult> results;

  // Summary fields populated when loaded from JSON (results list is empty in history)
  final int? _storedTotalWindows;
  final int? _storedNotOkCount;
  final Map<String, int>? _storedNoiseBreakdown;

  VehicleSession({
    required this.id,
    required this.vehicleName,
    required this.startTime,
    this.endTime,
    required this.results,
    int? storedTotalWindows,
    int? storedNotOkCount,
    Map<String, int>? storedNoiseBreakdown,
  })  : _storedTotalWindows = storedTotalWindows,
        _storedNotOkCount = storedNotOkCount,
        _storedNoiseBreakdown = storedNoiseBreakdown;

  int get totalWindows => _storedTotalWindows ?? results.length;
  int get notOkCount   => _storedNotOkCount ?? results.where((r) => r.isNotOk).length;
  int get okCount      => totalWindows - notOkCount;
  bool get isNotOk     => notOkCount > okCount;
  double get notOkRate => totalWindows > 0 ? notOkCount / totalWindows : 0;

  Map<String, int> get noiseBreakdown {
    if (_storedNoiseBreakdown != null) return _storedNoiseBreakdown!;
    final map = <String, int>{};
    for (final r in results.where((r) => r.isNotOk)) {
      map[r.noiseTypeName] = (map[r.noiseTypeName] ?? 0) + 1;
    }
    final sorted = Map.fromEntries(
        map.entries.toList()..sort((a, b) => b.value.compareTo(a.value)));
    return sorted;
  }

  // JSON serialisation for SharedPreferences
  Map<String, dynamic> toJson() => {
    'id':            id,
    'vehicleName':   vehicleName,
    'startTime':     startTime.toIso8601String(),
    'endTime':       endTime?.toIso8601String(),
    'totalWindows':  totalWindows,
    'notOkCount':    notOkCount,
    'isNotOk':       isNotOk,
    'noiseBreakdown': noiseBreakdown,
  };

  static VehicleSession fromJson(Map<String, dynamic> j) {
    final rawBreakdown = j['noiseBreakdown'] as Map<String, dynamic>?;
    return VehicleSession(
      id:          j['id'],
      vehicleName: j['vehicleName'],
      startTime:   DateTime.parse(j['startTime']),
      endTime:     j['endTime'] != null ? DateTime.parse(j['endTime']) : null,
      results:     [],
      storedTotalWindows:   j['totalWindows'] as int?,
      storedNotOkCount:     j['notOkCount'] as int?,
      storedNoiseBreakdown: rawBreakdown?.map((k, v) => MapEntry(k, (v as num).toInt())),
    );
  }
}
