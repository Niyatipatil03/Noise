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

  // Stored summary — populated when loaded from JSON (results list is empty)
  final int? _savedTotalWindows;
  final int? _savedNotOkCount;
  final bool? _savedIsNotOk;
  final Map<String, int>? _savedNoiseBreakdown;

  VehicleSession({
    required this.id,
    required this.vehicleName,
    required this.startTime,
    this.endTime,
    required this.results,
    int? savedTotalWindows,
    int? savedNotOkCount,
    bool? savedIsNotOk,
    Map<String, int>? savedNoiseBreakdown,
  })  : _savedTotalWindows = savedTotalWindows,
        _savedNotOkCount = savedNotOkCount,
        _savedIsNotOk = savedIsNotOk,
        _savedNoiseBreakdown = savedNoiseBreakdown;

  int get totalWindows => results.isNotEmpty ? results.length : (_savedTotalWindows ?? 0);
  int get notOkCount   => results.isNotEmpty ? results.where((r) => r.isNotOk).length : (_savedNotOkCount ?? 0);
  int get okCount      => totalWindows - notOkCount;
  bool get isNotOk     => results.isNotEmpty ? notOkCount > okCount : (_savedIsNotOk ?? false);
  double get notOkRate => totalWindows > 0 ? notOkCount / totalWindows : 0;

  Map<String, int> get noiseBreakdown {
    if (results.isEmpty && _savedNoiseBreakdown != null) return _savedNoiseBreakdown!;
    final map = <String, int>{};
    for (final r in results.where((r) => r.isNotOk)) {
      map[r.noiseTypeName] = (map[r.noiseTypeName] ?? 0) + 1;
    }
    return Map.fromEntries(
        map.entries.toList()..sort((a, b) => b.value.compareTo(a.value)));
  }

  Map<String, dynamic> toJson() => {
    'id':             id,
    'vehicleName':    vehicleName,
    'startTime':      startTime.toIso8601String(),
    'endTime':        endTime?.toIso8601String(),
    'totalWindows':   totalWindows,
    'notOkCount':     notOkCount,
    'isNotOk':        isNotOk,
    'noiseBreakdown': noiseBreakdown,
  };

  static VehicleSession fromJson(Map<String, dynamic> j) => VehicleSession(
    id:          j['id'],
    vehicleName: j['vehicleName'],
    startTime:   DateTime.parse(j['startTime']),
    endTime:     j['endTime'] != null ? DateTime.parse(j['endTime']) : null,
    results:     [],
    savedTotalWindows:   j['totalWindows'] as int?,
    savedNotOkCount:     j['notOkCount'] as int?,
    savedIsNotOk:        j['isNotOk'] as bool?,
    savedNoiseBreakdown: (j['noiseBreakdown'] as Map<String, dynamic>?)
        ?.map((k, v) => MapEntry(k, v as int)),
  );
}
