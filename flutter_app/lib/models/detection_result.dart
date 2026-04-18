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

  // Stored summary — populated from JSON when results list is empty
  final int _savedTotal;
  final int _savedNotOk;
  final Map<String, int> _savedBreakdown;

  VehicleSession({
    required this.id,
    required this.vehicleName,
    required this.startTime,
    this.endTime,
    required this.results,
    int savedTotal = 0,
    int savedNotOk = 0,
    Map<String, int>? savedBreakdown,
  })  : _savedTotal = savedTotal,
        _savedNotOk = savedNotOk,
        _savedBreakdown = savedBreakdown ?? {};

  // Use live results when available, fall back to stored summary after deserialisation
  int get totalWindows => results.isNotEmpty ? results.length : _savedTotal;
  int get notOkCount =>
      results.isNotEmpty ? results.where((r) => r.isNotOk).length : _savedNotOk;
  int get okCount    => totalWindows - notOkCount;
  bool get isNotOk   => notOkCount > okCount;
  double get notOkRate => totalWindows > 0 ? notOkCount / totalWindows : 0;

  Map<String, int> get noiseBreakdown {
    if (results.isEmpty) return _savedBreakdown;
    final map = <String, int>{};
    for (final r in results.where((r) => r.isNotOk)) {
      map[r.noiseTypeName] = (map[r.noiseTypeName] ?? 0) + 1;
    }
    return Map.fromEntries(
        map.entries.toList()..sort((a, b) => b.value.compareTo(a.value)));
  }

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
    final breakdown = <String, int>{};
    if (j['noiseBreakdown'] != null) {
      (j['noiseBreakdown'] as Map)
          .forEach((k, v) => breakdown[k.toString()] = (v as num).toInt());
    }
    return VehicleSession(
      id:           j['id'],
      vehicleName:  j['vehicleName'],
      startTime:    DateTime.parse(j['startTime']),
      endTime:      j['endTime'] != null ? DateTime.parse(j['endTime']) : null,
      results:      [],
      savedTotal:   (j['totalWindows'] as num?)?.toInt() ?? 0,
      savedNotOk:   (j['notOkCount']  as num?)?.toInt() ?? 0,
      savedBreakdown: breakdown,
    );
  }
}
