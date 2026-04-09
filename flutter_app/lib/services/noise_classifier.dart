import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'audio_processor.dart';
import '../models/detection_result.dart';

class NoiseClassifier {
  static const double threshold = 0.5;
  static const List<String> noiseTypeNames = [
    'OK', 'IP Noise', 'Sunroof Noise', 'Rear Noise',
    'Steering Noise', 'Door Noise', 'Glass Noise', 'Unknown Noise',
  ];

  Interpreter? _interpreter;
  final AudioProcessor _processor = AudioProcessor();

  Future<void> init() async {
    final options = InterpreterOptions()..threads = 4;
    ByteData byteData;
    try {
      byteData = await rootBundle.load('assets/bsr_detector.tflite');
    } catch (e) {
      throw Exception('Asset not found: $e');
    }
    try {
      _interpreter = Interpreter.fromBuffer(
        byteData.buffer.asUint8List(),
        options: options,
      );
    } catch (e) {
      throw Exception('Interpreter failed (size=${byteData.lengthInBytes}B): $e');
    }
  }

  DetectionResult classify(Uint8List pcm16Bytes) {
    final interp = _interpreter;
    if (interp == null) throw StateError('Model not loaded. Call init() first.');

    // Preprocess audio → spectrogram
    final specData = _processor.process(pcm16Bytes);

    // Input shape: [1, 128, 300, 1]
    final input = specData.reshape([1, AudioProcessor.nMels, AudioProcessor.tFrames, 1]);

    // Output buffers
    final binaryOut = List.filled(1, 0.0).reshape([1, 1]);
    final typeOut   = List.filled(noiseTypeNames.length, 0.0).reshape([1, noiseTypeNames.length]);

    interp.runForMultipleInputs(
      [input],
      {0: typeOut, 1: binaryOut},
    );

    final notOkProb  = (binaryOut[0][0] as double);
    final typeProbs  = (typeOut[0] as List).map((v) => (v as double)).toList();
    final topIdx     = typeProbs.indexOf(typeProbs.reduce((a, b) => a > b ? a : b));

    return DetectionResult(
      isNotOk          : notOkProb > threshold,
      notOkProbability : notOkProb,
      noiseTypeIndex   : topIdx,
      noiseTypeName    : noiseTypeNames[topIdx],
      noiseTypeProbs   : typeProbs,
      timestamp        : DateTime.now(),
    );
  }

  void dispose() => _interpreter?.close();
}
