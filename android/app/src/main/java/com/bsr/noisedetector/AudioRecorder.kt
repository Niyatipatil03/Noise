package com.bsr.noisedetector

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothHeadset
import android.bluetooth.BluetoothProfile
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * Manages audio capture from Bluetooth SCO mic (or built-in fallback).
 *
 * Produces a continuous stream of [NoiseClassifier.CLIP_SAMPLES]-length
 * Float32 PCM buffers at 16 kHz for classification.
 */
class AudioRecorder(private val context: Context) {

    companion object {
        private const val TAG = "AudioRecorder"
        private const val CHANNEL_CONFIG  = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT    = AudioFormat.ENCODING_PCM_FLOAT
        private const val SAMPLE_RATE     = NoiseClassifier.SAMPLE_RATE
        private const val CLIP_SAMPLES    = NoiseClassifier.CLIP_SAMPLES
        // Rolling buffer overlap: start new clip every HOP_DURATION samples
        private const val HOP_SAMPLES     = SAMPLE_RATE * 1   // 1 second hop
    }

    // ── State ─────────────────────────────────────────────────────────────────

    enum class State { IDLE, CONNECTING_BT, RECORDING, ERROR }

    private val _state = MutableStateFlow(State.IDLE)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _audioClips = MutableSharedFlow<FloatArray>(extraBufferCapacity = 4)
    val audioClips: SharedFlow<FloatArray> = _audioClips.asSharedFlow()

    private val _btConnected = MutableStateFlow(false)
    val bluetoothConnected: StateFlow<Boolean> = _btConnected.asStateFlow()

    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    // ── Bluetooth SCO ─────────────────────────────────────────────────────────

    private val btReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            when (intent.action) {
                AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED -> {
                    val state = intent.getIntExtra(
                        AudioManager.EXTRA_SCO_AUDIO_STATE,
                        AudioManager.SCO_AUDIO_STATE_ERROR,
                    )
                    when (state) {
                        AudioManager.SCO_AUDIO_STATE_CONNECTED -> {
                            Log.i(TAG, "Bluetooth SCO connected")
                            _btConnected.value = true
                            _state.value = State.RECORDING
                            startAudioRecord(useBluetooth = true)
                        }
                        AudioManager.SCO_AUDIO_STATE_DISCONNECTED -> {
                            Log.w(TAG, "Bluetooth SCO disconnected — falling back to internal mic")
                            _btConnected.value = false
                            if (_state.value == State.RECORDING) {
                                startAudioRecord(useBluetooth = false)
                            }
                        }
                    }
                }
            }
        }
    }

    private val btFilter = IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED)

    // ── Public API ────────────────────────────────────────────────────────────

    fun startRecording() {
        if (_state.value == State.RECORDING) return
        if (!hasAudioPermission()) {
            _state.value = State.ERROR
            return
        }
        context.registerReceiver(btReceiver, btFilter)
        _state.value = State.CONNECTING_BT

        // Attempt Bluetooth SCO; timeout → fall back to built-in mic after 3 s
        audioManager.startBluetoothSco()
        audioManager.isBluetoothScoOn = true

        scope.launch {
            delay(3_000)
            if (_state.value == State.CONNECTING_BT) {
                Log.w(TAG, "BT SCO timeout — using internal mic")
                _state.value = State.RECORDING
                startAudioRecord(useBluetooth = false)
            }
        }
    }

    fun stopRecording() {
        recordingJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        audioManager.stopBluetoothSco()
        audioManager.isBluetoothScoOn = false
        try { context.unregisterReceiver(btReceiver) } catch (_: Exception) {}
        _state.value = State.IDLE
        _btConnected.value = false
    }

    fun release() {
        stopRecording()
        scope.cancel()
    }

    // ── Internal recording ────────────────────────────────────────────────────

    private fun startAudioRecord(useBluetooth: Boolean) {
        // Cancel any previous recording job
        recordingJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()

        val audioSource = if (useBluetooth)
            MediaRecorder.AudioSource.VOICE_COMMUNICATION
        else
            MediaRecorder.AudioSource.MIC

        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val bufSize = maxOf(minBuf, CLIP_SAMPLES * 4)   // float32 = 4 bytes

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            _state.value = State.ERROR
            return
        }

        val record = AudioRecord(audioSource, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufSize)
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord init failed")
            _state.value = State.ERROR
            return
        }
        audioRecord = record
        record.startRecording()
        Log.i(TAG, "Recording started (bluetooth=$useBluetooth)")

        recordingJob = scope.launch {
            val rollingBuf = FloatArray(CLIP_SAMPLES)
            var rollingPos = 0     // samples written into rollingBuf so far

            val readBuf = FloatArray(HOP_SAMPLES)

            while (isActive) {
                val read = record.read(readBuf, 0, HOP_SAMPLES, AudioRecord.READ_BLOCKING)
                if (read <= 0) continue

                // Append to rolling buffer
                var srcIdx = 0
                while (srcIdx < read) {
                    val copyLen = minOf(read - srcIdx, CLIP_SAMPLES - rollingPos)
                    readBuf.copyInto(rollingBuf, rollingPos, srcIdx, srcIdx + copyLen)
                    rollingPos += copyLen
                    srcIdx += copyLen

                    if (rollingPos >= CLIP_SAMPLES) {
                        // Emit clip
                        _audioClips.emit(rollingBuf.copyOf())
                        // Slide window by HOP_SAMPLES
                        rollingBuf.copyInto(rollingBuf, 0, HOP_SAMPLES, CLIP_SAMPLES)
                        rollingPos = CLIP_SAMPLES - HOP_SAMPLES
                    }
                }
            }
        }
    }

    private fun hasAudioPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
}
