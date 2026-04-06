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
  String get statusText => isNotOk ? 'NOT OK' : 'OK';
}
