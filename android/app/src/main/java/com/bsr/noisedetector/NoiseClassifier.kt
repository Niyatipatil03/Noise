package com.bsr.noisedetector

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*

/**
 * Wraps the TFLite BSR noise detection model.
 *
 * The model expects a log-mel spectrogram of shape (1, N_MELS, T_FRAMES, 1)
 * and returns:
 *   binary_out  : FloatArray(1)     — probability of NOT_OK
 *   type_out    : FloatArray(8)     — noise-type probabilities
 */
class NoiseClassifier(context: Context) {

    companion object {
        private const val TAG = "NoiseClassifier"

        // Must match config.yaml
        const val SAMPLE_RATE    = 16000
        const val CLIP_DURATION  = 3       // seconds
        const val CLIP_SAMPLES   = SAMPLE_RATE * CLIP_DURATION   // 48 000
        const val N_MELS         = 128
        const val N_FFT          = 1024
        const val HOP_LENGTH     = 160
        const val WIN_LENGTH     = 400
        const val F_MIN          = 50.0f
        const val F_MAX          = 8000.0f
        const val TOP_DB         = 80.0f
        val T_FRAMES             = ceil(CLIP_SAMPLES.toDouble() / HOP_LENGTH).toInt()   // 300

        // Decision threshold
        const val BINARY_THRESHOLD = 0.5f

        // Labels (must match NOISE_TYPE_LABELS in bsr_dataset.py)
        val NOISE_TYPE_NAMES = arrayOf(
            "OK",
            "IP Noise",
            "Sunroof Noise",
            "Rear Noise",
            "Steering Noise",
            "Door Noise",
            "Glass Noise",
            "Unknown Noise",
        )

        private const val MODEL_ASSET = "bsr_detector.tflite"
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    init {
        loadModel(context)
    }

    // ── Model loading ─────────────────────────────────────────────────────────

    private fun loadModel(context: Context) {
        try {
            val modelBuffer = loadModelFile(context)
            val options = Interpreter.Options().apply {
                numThreads = 4
                // Try GPU delegate; fall back silently to CPU
                try {
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate!!)
                } catch (e: Exception) {
                    Log.w(TAG, "GPU delegate unavailable, using CPU: ${e.message}")
                    gpuDelegate = null
                }
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "Model loaded. Input: ${interpreter!!.getInputTensor(0).shape().contentToString()}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model", e)
            throw RuntimeException("Cannot load BSR model from assets/$MODEL_ASSET", e)
        }
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val afd = context.assets.openFd(MODEL_ASSET)
        FileInputStream(afd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    // ── Audio preprocessing ───────────────────────────────────────────────────

    /**
     * Convert raw PCM float32 waveform (16 kHz mono) to log-mel spectrogram ByteBuffer.
     * Mirrors extract_spectrogram_for_model() in audio_features.py.
     */
    fun preprocessAudio(waveform: FloatArray): ByteBuffer {
        val padded = padOrTruncate(waveform, CLIP_SAMPLES)
        val normalised = peakNormalise(padded)

        // ── STFT → power spectrum ─────────────────────────────────────────────
        val numFrames = ceil(CLIP_SAMPLES.toDouble() / HOP_LENGTH).toInt()
        val fftBins   = N_FFT / 2 + 1   // 513
        val powerSpec = Array(numFrames) { FloatArray(fftBins) }

        val window = hanningWindow(N_FFT)

        for (frame in 0 until numFrames) {
            val start = frame * HOP_LENGTH
            val frame_buf = FloatArray(N_FFT)
            for (k in 0 until N_FFT) {
                val idx = start + k
                frame_buf[k] = if (idx < normalised.size) normalised[idx] * window[k] else 0f
            }
            val fft = realFFT(frame_buf)
            for (b in 0 until fftBins) {
                val re = fft[2 * b]
                val im = fft[2 * b + 1]
                powerSpec[frame][b] = re * re + im * im
            }
        }

        // ── Mel filterbank ────────────────────────────────────────────────────
        val melFilters = buildMelFilterbank(N_MELS, fftBins, SAMPLE_RATE.toFloat(), F_MIN, F_MAX)

        // ── Apply filters → log → normalise ──────────────────────────────────
        val logMel = Array(N_MELS) { FloatArray(T_FRAMES) }
        for (mel in 0 until N_MELS) {
            for (t in 0 until min(numFrames, T_FRAMES)) {
                var energy = 0f
                for (b in 0 until fftBins) {
                    energy += powerSpec[t][b] * melFilters[mel][b]
                }
                logMel[mel][t] = ln(energy.coerceAtLeast(1e-10f))
            }
        }

        // Convert dB with top_db limiting
        var maxVal = Float.NEGATIVE_INFINITY
        for (mel in 0 until N_MELS)
            for (t in 0 until T_FRAMES)
                if (logMel[mel][t] > maxVal) maxVal = logMel[mel][t]

        for (mel in 0 until N_MELS)
            for (t in 0 until T_FRAMES)
                logMel[mel][t] = maxOf(logMel[mel][t], maxVal - TOP_DB)

        // Zero-mean unit-variance normalisation
        var mean = 0f; var count = 0
        for (mel in 0 until N_MELS) for (t in 0 until T_FRAMES) { mean += logMel[mel][t]; count++ }
        mean /= count
        var variance = 0f
        for (mel in 0 until N_MELS) for (t in 0 until T_FRAMES) variance += (logMel[mel][t] - mean).pow(2)
        val std = sqrt(variance / count) + 1e-9f

        // Pack into ByteBuffer: shape (1, N_MELS, T_FRAMES, 1)
        val buf = ByteBuffer.allocateDirect(4 * N_MELS * T_FRAMES)
            .order(ByteOrder.nativeOrder())
        for (mel in 0 until N_MELS)
            for (t in 0 until T_FRAMES)
                buf.putFloat((logMel[mel][t] - mean) / std)
        buf.rewind()
        return buf
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    data class Result(
        val isNotOk: Boolean,
        val notOkProbability: Float,
        val noiseTypeIndex: Int,
        val noiseTypeName: String,
        val noiseTypeProbs: FloatArray,
    )

    fun classify(waveform: FloatArray): Result {
        val interp = interpreter ?: throw IllegalStateException("Model not loaded")

        val inputBuf = preprocessAudio(waveform)

        // Allocate output buffers
        val binaryOut = Array(1) { FloatArray(1) }
        val typeOut   = Array(1) { FloatArray(NOISE_TYPE_NAMES.size) }

        val outputs = mapOf(
            0 to binaryOut,
            1 to typeOut,
        )

        interp.runForMultipleInputsOutputs(arrayOf(inputBuf), outputs)

        val notOkProb    = binaryOut[0][0]
        val typeProbs    = typeOut[0]
        val topTypeIdx   = typeProbs.indices.maxByOrNull { typeProbs[it] } ?: 0
        val topTypeName  = NOISE_TYPE_NAMES.getOrElse(topTypeIdx) { "Unknown" }

        return Result(
            isNotOk          = notOkProb > BINARY_THRESHOLD,
            notOkProbability = notOkProb,
            noiseTypeIndex   = topTypeIdx,
            noiseTypeName    = topTypeName,
            noiseTypeProbs   = typeProbs,
        )
    }

    // ── DSP helpers ───────────────────────────────────────────────────────────

    private fun padOrTruncate(wav: FloatArray, target: Int): FloatArray {
        if (wav.size == target) return wav
        if (wav.size > target) return wav.copyOf(target)
        val result = FloatArray(target)
        var pos = 0
        while (pos < target) {
            val copyLen = minOf(wav.size, target - pos)
            wav.copyInto(result, pos, 0, copyLen)
            pos += copyLen
        }
        return result
    }

    private fun peakNormalise(wav: FloatArray): FloatArray {
        val peak = wav.maxOfOrNull { abs(it) } ?: 1f
        val safeDiv = if (peak < 1e-9f) 1f else peak
        return FloatArray(wav.size) { wav[it] / safeDiv }
    }

    private fun hanningWindow(size: Int): FloatArray =
        FloatArray(size) { n -> 0.5f * (1f - cos(2.0 * PI * n / (size - 1)).toFloat()) }

    /**
     * Real-valued Cooley-Tukey FFT.
     * Returns interleaved [re0, im0, re1, im1, …] of length N.
     */
    private fun realFFT(x: FloatArray): FloatArray {
        val n = x.size
        val out = FloatArray(n * 2)
        // Copy real parts; imaginary = 0
        for (i in 0 until n) { out[2 * i] = x[i]; out[2 * i + 1] = 0f }

        // Bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                var tmp = out[2 * i]; out[2 * i] = out[2 * j]; out[2 * j] = tmp
                tmp = out[2 * i + 1]; out[2 * i + 1] = out[2 * j + 1]; out[2 * j + 1] = tmp
            }
        }

        // Butterfly
        var len = 2
        while (len <= n) {
            val wRe = cos(-2.0 * PI / len).toFloat()
            val wIm = sin(-2.0 * PI / len).toFloat()
            var start = 0
            while (start < n) {
                var uRe = 1f; var uIm = 0f
                for (k in 0 until len / 2) {
                    val evenRe = out[2 * (start + k)]
                    val evenIm = out[2 * (start + k) + 1]
                    val oddRe  = out[2 * (start + k + len / 2)]
                    val oddIm  = out[2 * (start + k + len / 2) + 1]
                    val tRe = uRe * oddRe - uIm * oddIm
                    val tIm = uRe * oddIm + uIm * oddRe
                    out[2 * (start + k)]               = evenRe + tRe
                    out[2 * (start + k) + 1]           = evenIm + tIm
                    out[2 * (start + k + len / 2)]     = evenRe - tRe
                    out[2 * (start + k + len / 2) + 1] = evenIm - tIm
                    val newURe = uRe * wRe - uIm * wIm
                    uIm = uRe * wIm + uIm * wRe; uRe = newURe
                }
                start += len
            }
            len = len shl 1
        }
        return out
    }

    private fun hzToMel(hz: Float): Float = 2595f * log10(1f + hz / 700f)
    private fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1f)

    private fun buildMelFilterbank(
        numMels: Int, numBins: Int, sr: Float, fMin: Float, fMax: Float,
    ): Array<FloatArray> {
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = FloatArray(numMels + 2) { i ->
            melToHz(melMin + i * (melMax - melMin) / (numMels + 1))
        }
        val binPoints = FloatArray(numMels + 2) { i ->
            floor(melPoints[i] / sr * (2 * (numBins - 1))).toInt().toFloat()
        }
        return Array(numMels) { m ->
            FloatArray(numBins) { b ->
                when {
                    b < binPoints[m]     -> 0f
                    b <= binPoints[m + 1] -> (b - binPoints[m]) / (binPoints[m + 1] - binPoints[m])
                    b <= binPoints[m + 2] -> (binPoints[m + 2] - b) / (binPoints[m + 2] - binPoints[m + 1])
                    else                  -> 0f
                }
            }
        }
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
