import 'dart:async';
import 'dart:typed_data';
import 'package:record/record.dart';

/// Streams rolling 3-second PCM-16 audio windows at 16 kHz mono.
class AudioRecorderService {
  static const int sampleRate  = 16000;
  static const int clipSeconds = 3;
  static const int hopSeconds  = 1;

  static const int clipSamples = sampleRate * clipSeconds; // 48000
  static const int hopSamples  = sampleRate * hopSeconds;  // 16000
  // PCM16 = 2 bytes per sample
  static const int clipBytes   = clipSamples * 2;
  static const int hopBytes    = hopSamples * 2;

  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _sub;
  final _clips = StreamController<Uint8List>.broadcast();

  Stream<Uint8List> get clips => _clips.stream;
  bool get isRecording => _sub != null;

  Future<bool> hasPermission() => _recorder.hasPermission();

  Future<void> start() async {
    final stream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: sampleRate,
        numChannels: 1,
        bitRate: 16 * sampleRate,
      ),
    );

    // Rolling buffer
    final buffer = <int>[];

    _sub = stream.listen((chunk) {
      buffer.addAll(chunk);

      while (buffer.length >= clipBytes) {
        // Emit current clip
        final clip = Uint8List.fromList(buffer.sublist(0, clipBytes));
        _clips.add(clip);
        // Slide window by hop
        buffer.removeRange(0, hopBytes);
      }
    });
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
    await _recorder.stop();
  }

  void dispose() {
    stop();
    _recorder.dispose();
    _clips.close();
  }
}
