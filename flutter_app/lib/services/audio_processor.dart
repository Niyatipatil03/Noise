import 'dart:math';
import 'dart:typed_data';

/// Converts raw PCM audio → log-mel spectrogram matching the Python training pipeline.
///
/// Config must match config.yaml:
///   sample_rate : 16000
///   n_mels      : 128
///   n_fft       : 1024
///   hop_length  : 160
///   clip_samples: 48000  (3 s × 16 kHz)
class AudioProcessor {
  static const int sampleRate   = 16000;
  static const int clipSamples  = 48000;   // 3 seconds
  static const int nMels        = 128;
  static const int nFft         = 1024;
  static const int hopLength    = 160;
  static const int winLength    = 400;
  static const double fMin      = 50.0;
  static const double fMax      = 8000.0;
  static const double topDb     = 80.0;

  // T = ceil(clipSamples / hopLength) = ceil(48000/160) = 300
  static const int tFrames = 300;

  late final List<double> _hanningWindow;
  late final List<List<double>> _melFilterbank;

  AudioProcessor() {
    _hanningWindow  = _buildHanningWindow(nFft);
    _melFilterbank  = _buildMelFilterbank(nMels, nFft ~/ 2 + 1, sampleRate.toDouble(), fMin, fMax);
  }

  /// Main entry point.
  /// [pcm16] : raw PCM-16 bytes from the record plugin
  /// Returns Float32List of shape [1 * nMels * tFrames * 1] (flattened, row-major)
  Float32List process(Uint8List pcm16) {
    final waveform = _pcm16ToFloat(pcm16);
    final padded   = _padOrTruncate(waveform, clipSamples);
    final normed   = _peakNormalise(padded);
    final logMel   = _logMelSpectrogram(normed);
    final normedSpec = _normaliseSpec(logMel);
    return _flatten(normedSpec);
  }

  // ── PCM16 → float32 ───────────────────────────────────────────────────────

  List<double> _pcm16ToFloat(Uint8List bytes) {
    final shorts = Int16List.view(bytes.buffer);
    return List.generate(shorts.length, (i) => shorts[i] / 32768.0);
  }

  // ── Pad / truncate ────────────────────────────────────────────────────────

  List<double> _padOrTruncate(List<double> wav, int target) {
    if (wav.length == target) return wav;
    if (wav.length > target)  return wav.sublist(0, target);
    final out = List<double>.filled(target, 0.0);
    int pos = 0;
    while (pos < target) {
      final copy = min(wav.length, target - pos);
      for (int i = 0; i < copy; i++) out[pos + i] = wav[i];
      pos += copy;
    }
    return out;
  }

  // ── Peak normalise ────────────────────────────────────────────────────────

  List<double> _peakNormalise(List<double> wav) {
    double peak = wav.fold(0.0, (m, v) => max(m, v.abs()));
    if (peak < 1e-9) peak = 1.0;
    return wav.map((v) => v / peak).toList();
  }

  // ── Hanning window ────────────────────────────────────────────────────────

  List<double> _buildHanningWindow(int size) =>
      List.generate(size, (n) => 0.5 * (1.0 - cos(2.0 * pi * n / (size - 1))));

  // ── Real FFT (Cooley-Tukey) ───────────────────────────────────────────────

  /// Returns interleaved [re0, im0, re1, im1, …] of length n*2
  List<double> _realFft(List<double> x) {
    final n   = x.length;
    final out = List<double>.filled(n * 2, 0.0);
    for (int i = 0; i < n; i++) out[2 * i] = x[i];

    // Bit-reversal
    int j = 0;
    for (int i = 1; i < n; i++) {
      int bit = n >> 1;
      while (j & bit != 0) { j ^= bit; bit >>= 1; }
      j ^= bit;
      if (i < j) {
        double tmp;
        tmp = out[2*i]; out[2*i] = out[2*j]; out[2*j] = tmp;
        tmp = out[2*i+1]; out[2*i+1] = out[2*j+1]; out[2*j+1] = tmp;
      }
    }

    // Butterfly
    int len = 2;
    while (len <= n) {
      final wRe = cos(-2.0 * pi / len);
      final wIm = sin(-2.0 * pi / len);
      for (int start = 0; start < n; start += len) {
        double uRe = 1.0, uIm = 0.0;
        for (int k = 0; k < len ~/ 2; k++) {
          final eR = out[2*(start+k)],     eI = out[2*(start+k)+1];
          final oR = out[2*(start+k+len~/2)], oI = out[2*(start+k+len~/2)+1];
          final tR = uRe*oR - uIm*oI,     tI = uRe*oI + uIm*oR;
          out[2*(start+k)]           = eR + tR;
          out[2*(start+k)+1]         = eI + tI;
          out[2*(start+k+len~/2)]    = eR - tR;
          out[2*(start+k+len~/2)+1]  = eI - tI;
          final newURe = uRe*wRe - uIm*wIm;
          uIm = uRe*wIm + uIm*wRe; uRe = newURe;
        }
      }
      len <<= 1;
    }
    return out;
  }

  // ── Mel filterbank ────────────────────────────────────────────────────────

  double _hzToMel(double hz) => 2595.0 * log(1.0 + hz / 700.0) / ln10;
  double _melToHz(double mel) => 700.0 * (pow(10.0, mel / 2595.0) - 1.0);

  List<List<double>> _buildMelFilterbank(int nMels, int nBins, double sr, double fMin, double fMax) {
    final melMin = _hzToMel(fMin);
    final melMax = _hzToMel(fMax);
    final melPts = List.generate(nMels + 2,
        (i) => _melToHz(melMin + i * (melMax - melMin) / (nMels + 1)));
    final binPts = melPts.map((f) => (f / sr * 2 * (nBins - 1)).floor().toDouble()).toList();

    return List.generate(nMels, (m) {
      return List.generate(nBins, (b) {
        final lo = binPts[m], mid = binPts[m+1], hi = binPts[m+2];
        if (b >= lo && b <= mid && mid > lo) return (b - lo) / (mid - lo);
        if (b > mid && b <= hi && hi > mid)  return (hi - b) / (hi - mid);
        return 0.0;
      });
    });
  }

  // ── Log-mel spectrogram ───────────────────────────────────────────────────

  /// Returns [nMels][tFrames] log-mel spectrogram
  List<List<double>> _logMelSpectrogram(List<double> wav) {
    final nBins   = nFft ~/ 2 + 1;
    final nFrames = (wav.length / hopLength).ceil();

    final logMel = List.generate(nMels, (_) => List<double>.filled(tFrames, 0.0));

    for (int t = 0; t < min(nFrames, tFrames); t++) {
      final start    = t * hopLength;
      final frameBuf = List<double>.filled(nFft, 0.0);
      for (int k = 0; k < nFft; k++) {
        final idx = start + k;
        frameBuf[k] = (idx < wav.length) ? wav[idx] * _hanningWindow[k] : 0.0;
      }

      final fft   = _realFft(frameBuf);
      final power = List<double>.generate(nBins, (b) {
        final re = fft[2*b], im = fft[2*b+1];
        return re*re + im*im;
      });

      for (int m = 0; m < nMels; m++) {
        double energy = 0.0;
        for (int b = 0; b < nBins; b++) energy += power[b] * _melFilterbank[m][b];
        logMel[m][t] = log(max(energy, 1e-10));
      }
    }

    // Convert to dB with top_db limiting
    double maxVal = double.negativeInfinity;
    for (int m = 0; m < nMels; m++)
      for (int t = 0; t < tFrames; t++)
        if (logMel[m][t] > maxVal) maxVal = logMel[m][t];

    for (int m = 0; m < nMels; m++)
      for (int t = 0; t < tFrames; t++)
        logMel[m][t] = max(logMel[m][t], maxVal - topDb);

    return logMel;
  }

  // ── Zero-mean unit-variance normalisation ─────────────────────────────────

  List<List<double>> _normaliseSpec(List<List<double>> spec) {
    double sum = 0.0, count = 0.0;
    for (final row in spec) for (final v in row) { sum += v; count++; }
    final mean = sum / count;

    double varSum = 0.0;
    for (final row in spec) for (final v in row) varSum += (v - mean) * (v - mean);
    final std = sqrt(varSum / count) + 1e-9;

    return List.generate(nMels, (m) =>
        List.generate(tFrames, (t) => (spec[m][t] - mean) / std));
  }

  // ── Flatten to Float32List for TFLite ─────────────────────────────────────

  Float32List _flatten(List<List<double>> spec) {
    // Shape: [1, nMels, tFrames, 1]
    final out = Float32List(nMels * tFrames);
    int idx = 0;
    for (int m = 0; m < nMels; m++)
      for (int t = 0; t < tFrames; t++)
        out[idx++] = spec[m][t].toFloat();
    return out;
  }
}

extension on double {
  double toFloat() => this;
}
