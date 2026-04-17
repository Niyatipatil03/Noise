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

  // Saved summary values restored from JSON (override computed values)
  final int? _savedTotal;
  final int? _savedNotOk;
  final bool? _savedIsNotOk;
  final Map<String, int>? _savedBreakdown;

  VehicleSession({
    required this.id,
    required this.vehicleName,
    required this.startTime,
    this.endTime,
    required this.results,
    int? savedTotal,
    int? savedNotOk,
    bool? savedIsNotOk,
    Map<String, int>? savedBreakdown,
  })  : _savedTotal = savedTotal,
        _savedNotOk = savedNotOk,
        _savedIsNotOk = savedIsNotOk,
        _savedBreakdown = savedBreakdown;

  int get totalWindows  => _savedTotal ?? results.length;
  int get notOkCount    => _savedNotOk ?? results.where((r) => r.isNotOk).length;
  int get okCount       => totalWindows - notOkCount;
  bool get isNotOk      => _savedIsNotOk ?? notOkCount > okCount;
  double get notOkRate  => totalWindows > 0 ? notOkCount / totalWindows : 0;

  Map<String, int> get noiseBreakdown {
    if (_savedBreakdown != null) return _savedBreakdown!;
    final map = <String, int>{};
    for (final r in results.where((r) => r.isNotOk)) {
      map[r.noiseTypeName] = (map[r.noiseTypeName] ?? 0) + 1;
    }
    return Map.fromEntries(
        map.entries.toList()..sort((a, b) => b.value.compareTo(a.value)));
  }

  // JSON serialisation for SharedPreferences
  Map<String, dynamic> toJson() => {
    'id':          id,
    'vehicleName': vehicleName,
    'startTime':   startTime.toIso8601String(),
    'endTime':     endTime?.toIso8601String(),
    'totalWindows': totalWindows,
    'notOkCount':  notOkCount,
    'isNotOk':     isNotOk,
    'noiseBreakdown': noiseBreakdown,
  };

  static VehicleSession fromJson(Map<String, dynamic> j) => VehicleSession(
    id:          j['id'],
    vehicleName: j['vehicleName'],
    startTime:   DateTime.parse(j['startTime']),
    endTime:     j['endTime'] != null ? DateTime.parse(j['endTime']) : null,
    results:     [],
    savedTotal:    j['totalWindows'] as int?,
    savedNotOk:    j['notOkCount'] as int?,
    savedIsNotOk:  j['isNotOk'] as bool?,
    savedBreakdown: (j['noiseBreakdown'] as Map?)
        ?.map((k, v) => MapEntry(k as String, (v as num).toInt())),
  );
}
